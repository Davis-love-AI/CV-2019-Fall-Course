# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder, BoxCoder2
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_riou
from maskrcnn_benchmark.layers import riou


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        if cfg.ROTATE:
            if cfg.R2CNN:
                match_quality_matrix = boxlist_riou(proposal, target)
            else:
                match_quality_matrix = boxlist_riou(proposal, target)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # Fast RCNN only need "labels" field for selecting the targets
            target = target.copy_with_fields(["labels", "xyxy", "xywht", "xywht1"])
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_targets = target[matched_idxs.clamp(min=0)]
            matched_targets.add_field("matched_idxs", matched_idxs)
        else:
            match_quality_matrix = boxlist_iou(target, proposal)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # Fast RCNN only need "labels" field for selecting the targets
            target = target.copy_with_fields("labels")
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_targets = target[matched_idxs.clamp(min=0)]
            matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            proposals_per_image.add_field("matched_idxs", matched_idxs)
            # compute regression targets
            if cfg.ROTATE:
                if cfg.R2CNN:
                    regression_targets_per_image_r = self.box_coder.encode(
                        matched_targets.extra_fields["xywht"], proposals_per_image.extra_fields["xywht"]
                    )
                    regression_targets_per_image_h = self.box_coder.encode(
                        matched_targets.extra_fields["xyxy"], proposals_per_image.extra_fields["xyxy"]
                    )
                    regression_targets_per_image = torch.cat((regression_targets_per_image_r, regression_targets_per_image_h), dim=1)

                    if cfg.MULTI_REG:
                        regression_targets_per_image_r_1 = self.box_coder.encode(
                            matched_targets.extra_fields["xywht1"], proposals_per_image.extra_fields["xywht"]
                        )
                        regression_targets_per_image = torch.cat((regression_targets_per_image, regression_targets_per_image_r_1), dim=1)

                    labels.append(labels_per_image)
                    regression_targets.append(regression_targets_per_image)

                    return labels, regression_targets

                else:
                    regression_targets_per_image = self.box_coder.encode(
                        matched_targets.extra_fields["xywht"], proposals_per_image.extra_fields["xywht"]
                    )
                    if cfg.MULTI_REG:
                        regression_targets_per_image_1 = self.box_coder.encode(
                            matched_targets.extra_fields["xywht1"], proposals_per_image.extra_fields["xywht"]
                        )
                        regression_targets_per_image = torch.cat((regression_targets_per_image, regression_targets_per_image_1), dim=1)

                    labels.append(labels_per_image)
                    regression_targets.append(regression_targets_per_image)

                    return labels, regression_targets

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets = self.prepare_targets(proposals, targets)
        #         sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        if cfg.BALANCE_GT:
            gt_number = len(labels) - cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels, gt_number=gt_number)
        else:
            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression, class_logits_h=None, box_regression_h=None, targets=None):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        if cfg.R2CNN:
            class_logits_h = cat(class_logits_h, dim=0)
            box_regression_h = cat(box_regression_h, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )
        if cfg.R2CNN:
            classification_loss_h = F.cross_entropy(class_logits_h, labels)

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        elif cfg.ROTATE:
            if cfg.R2CNN:
                map_inds4 = 4 * labels_pos[:, None] + torch.tensor(
                    [0, 1, 2, 3], device=device)

            map_inds = 5 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3, 4], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)
        ###R2CNN
        if cfg.R2CNN:
            if cfg.FACTOR_IOU_LOSS:
                print("R2CNN IOU_FACOT NONE!!!")
                exit()
                proposal_bbox = cat([proposal.bbox for proposal in proposals], dim=0)
                target_bbox = cat([target.bbox for target in targets], dim=0)
                matched_idx = cat([proposal.get_field("matched_idxs") for proposal in proposals], dim=0)

                proposal_bbox_map = proposal_bbox[sampled_pos_inds_subset]
                matched_idx_map = matched_idx[sampled_pos_inds_subset][:, None]
                target_bbox_map = target_bbox[matched_idx_map].squeeze(1)

                match_quality_matrix = riou(proposal_bbox_map, target_bbox_map)
                match_quality_matrix = match_quality_matrix.reshape(len(proposal_bbox_map), len(target_bbox_map)).transpose(0, 1)

                iou = match_quality_matrix.cuda().float() * torch.eye(match_quality_matrix.size(0), match_quality_matrix.size(0),
                                                                      dtype=torch.float).cuda()

                index = torch.nonzero(iou)
                iou = iou[index[:, 0], index[:, 1]][:, None]

                box_loss = smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset[:, None], map_inds],
                    regression_targets[sampled_pos_inds_subset],
                    size_average=False,
                    beta=1,
                    iou=iou
                )
                box_loss = box_loss / labels.numel()
                return cfg.MODEL.ROI_HEADS.CLS_WEIGHT * classification_loss, cfg.MODEL.ROI_HEADS.LOC_WEIGHT * box_loss

            elif cfg.MULTI_REG:
                regression_targets_r_1 = regression_targets[:, :5]
                regression_targets_r_2 = regression_targets[:, -5:]

                regression_targets_h = regression_targets[:, 5:-5]
                box_loss1 = smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset[:, None], map_inds],
                    regression_targets_r_1[sampled_pos_inds_subset],
                    size_average=False,
                    beta=1,
                    multi_reg=True
                )

                box_loss2 = smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset[:, None], map_inds],
                    regression_targets_r_2[sampled_pos_inds_subset],
                    size_average=False,
                    beta=1,
                    multi_reg=True
                )

                box_p1 = torch.exp(- box_loss1.sum(1)[:, None])
                box_p2 = torch.exp(- box_loss2.sum(1)[:, None])

                box_w1, box_w2 = 1 / (1 - box_p1), 1 / (1 - box_p2)

                box_multi_loss = - torch.log(
                    #                torch.max(box_p1, box_p2)
                    #                 1 - (1 - box_p1) * (1 - box_p2) + 1e-7
                    (box_p1 * box_w1 + box_p2 * box_w2) / (box_w1 + box_w2)
                )

                box_h_loss = smooth_l1_loss(
                    box_regression_h[sampled_pos_inds_subset[:, None], map_inds4],
                    regression_targets_h[sampled_pos_inds_subset],
                    size_average=False,
                    beta=1,
                )

                box_multi_loss = box_multi_loss.sum() / labels.numel()
                box_h_loss = box_h_loss / labels.numel()

                return cfg.MODEL.ROI_HEADS.CLS_WEIGHT * classification_loss, cfg.MODEL.ROI_HEADS.LOC_WEIGHT * box_multi_loss, cfg.MODEL.ROI_HEADS.CLS_WEIGHT * classification_loss_h, cfg.MODEL.ROI_HEADS.LOC_WEIGHT * box_h_loss, box_loss1.clone().detach().sum() / labels.numel(), box_loss2.clone().detach().sum() / labels.numel()

            else:
                regression_targets_r = regression_targets[:, :5]
                regression_targets_h = regression_targets[:, 5:]

                box_r_loss = smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset[:, None], map_inds],
                    regression_targets_r[sampled_pos_inds_subset],
                    size_average=False,
                    beta=1,
                )

                box_h_loss = smooth_l1_loss(
                    box_regression_h[sampled_pos_inds_subset[:, None], map_inds4],
                    regression_targets_h[sampled_pos_inds_subset],
                    size_average=False,
                    beta=1,
                )
                box_r_loss = box_r_loss / labels.numel()
                box_h_loss = box_h_loss / labels.numel()

                return cfg.MODEL.ROI_HEADS.CLS_WEIGHT * classification_loss, cfg.MODEL.ROI_HEADS.LOC_WEIGHT * box_r_loss, cfg.MODEL.ROI_HEADS.CLS_WEIGHT * classification_loss_h, cfg.MODEL.ROI_HEADS.LOC_WEIGHT * box_h_loss
        ###RCNN
        else:
            if cfg.FACTOR_IOU_LOSS:
                proposal_bbox = cat([proposal.bbox for proposal in proposals], dim=0)
                target_bbox = cat([target.bbox for target in targets], dim=0)
                matched_idx = cat([proposal.get_field("matched_idxs") for proposal in proposals], dim=0)

                proposal_bbox_map = proposal_bbox[sampled_pos_inds_subset]
                matched_idx_map = matched_idx[sampled_pos_inds_subset][:, None]
                target_bbox_map = target_bbox[matched_idx_map].squeeze(1)

                match_quality_matrix = riou(proposal_bbox_map, target_bbox_map)
                match_quality_matrix = match_quality_matrix.reshape(len(proposal_bbox_map), len(target_bbox_map)).transpose(0, 1)

                iou = match_quality_matrix.cuda().float() * torch.eye(match_quality_matrix.size(0), match_quality_matrix.size(0),
                                                                      dtype=torch.float).cuda()

                index = torch.nonzero(iou)
                iou = iou[index[:, 0], index[:, 1]][:, None]

                box_loss = smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset[:, None], map_inds],
                    regression_targets[sampled_pos_inds_subset],
                    size_average=False,
                    beta=1,
                    iou=iou
                )
                box_loss = box_loss / labels.numel()
                return cfg.MODEL.ROI_HEADS.CLS_WEIGHT * classification_loss, cfg.MODEL.ROI_HEADS.LOC_WEIGHT * box_loss

            elif cfg.MULTI_REG:
                # 角度周期映射
                # dt = box_regression[:, 4]
                # dt1 = torch.where(dt >= 1.5708, dt % -3.1415926, dt)
                # dt2 = torch.where(dt1 <= - 1.5708, dt1 % 3.1415926, dt1)
                # box_regression[:, 4] = dt2
                regression_targets1 = regression_targets[:, :5]
                regression_targets2 = regression_targets[:, 5:]
                box_loss1 = smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset[:, None], map_inds],
                    regression_targets1[sampled_pos_inds_subset],
                    size_average=False,
                    beta=1,
                    multi_reg=True
                )

                box_loss2 = smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset[:, None], map_inds],
                    regression_targets2[sampled_pos_inds_subset],
                    size_average=False,
                    beta=1,
                    multi_reg=True
                )

                box_p1 = torch.exp(- box_loss1.sum(1)[:, None])
                box_p2 = torch.exp(- box_loss2.sum(1)[:, None])

                box_w1, box_w2 = 1 / (1 - box_p1), 1 / (1 - box_p2)

                box_multi_loss = - torch.log(
                    #                torch.max(box_p1, box_p2)
                    #                 1 - (1 - box_p1) * (1 - box_p2) + 1e-7
                    (box_p1 * box_w1 + box_p2 * box_w2) / (box_w1 + box_w2)
                )
                box_multi_loss = box_multi_loss.sum() / labels.numel()

                return cfg.MODEL.ROI_HEADS.CLS_WEIGHT * classification_loss, cfg.MODEL.ROI_HEADS.LOC_WEIGHT * box_multi_loss, box_loss1.clone().detach().sum() / labels.numel(), box_loss2.clone().detach().sum() / labels.numel()

            else:
                box_loss = smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset[:, None], map_inds],
                    regression_targets[sampled_pos_inds_subset],
                    size_average=False,
                    beta=1,
                )
                box_loss = box_loss / labels.numel()

                return cfg.MODEL.ROI_HEADS.CLS_WEIGHT * classification_loss, cfg.MODEL.ROI_HEADS.LOC_WEIGHT * box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder2(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
