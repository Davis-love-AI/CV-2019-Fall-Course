import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_retinanet_postprocessor
from .loss import make_retinanet_loss_evaluator
from ..anchor_generator import make_anchor_generator_retinanet

from maskrcnn_benchmark.modeling.box_coder import BoxCoder

from maskrcnn_benchmark.layers import DFConv2d


class RetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
                      * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        if cfg.ROTATE:
            self.bbox_pred = nn.Conv2d(
                in_channels, num_anchors * 5, kernel_size=3, stride=1,
                padding=1
            )
        else:
            self.bbox_pred = nn.Conv2d(
                in_channels, num_anchors * 4, kernel_size=3, stride=1,
                padding=1
            )

        # Initialization
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits,
                        self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # retinanet_bias_init
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
            logits.append(self.cls_logits(self.cls_tower(feature)))
        return logits, bbox_reg


class DCN_RetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(DCN_RetinaNetHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        self.factor = cfg.MODEL.DCN.BRANCH_FACTOR
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
                      * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE

        self.num_branch = num_anchors
        cls_branch = []

        for branch in range(self.num_branch):
            # first 256-32 dcn
            for i in range(cfg.MODEL.RETINANET.NUM_CONVS - 1):
                cls_branch.append(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
                cls_branch.append(nn.ReLU())

            cls_branch.append(
                DFConv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    with_modulated_dcn=False,
                    need_offset=True
                )
            )
            cls_branch.append(nn.ReLU())
            # second-forth 32-32 dcn
            #             for i in range(cfg.MODEL.RETINANET.NUM_CONVS - 1):
            #                 cls_branch.append(
            #                     DFConv2d(
            #                         in_channels // self.factor,
            #                         in_channels // self.factor,
            #                         kernel_size=3,
            #                         stride=1,
            #                         with_modulated_dcn=False,
            #                         need_offset=True
            #                     )
            #                 )
            #                 cls_branch.append(nn.ReLU())
            # cls_logits
            cls_branch.append(
                nn.Conv2d(
                    in_channels, 1 * num_classes, kernel_size=3, stride=1,
                    padding=1
                )
            )
            self.add_module('cls_branch_{:d}'.format(branch), nn.Sequential(*cls_branch))

            # Initialization
            m = getattr(self, "cls_branch_" + str(branch))
            for module in m:
                if isinstance(module, nn.Conv2d):
                    torch.nn.init.normal_(module.weight, std=0.01)
                    torch.nn.init.constant_(module.bias, 0)
            cls_branch = []

        # No initialization because DFConv2d has been initialization in the DFConv2d.py

        # retinanet_bias_init
        for branch in range(self.num_branch):
            # the last one is cls_logits
            m = getattr(self, "cls_branch_" + str(branch))
            prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(m[-1].bias, bias_value)

    def feacture_anchor_match(self, offset, scale_transform):
        dx1 = offset[:, 0, :, :][:, None, :, :]
        dy1 = offset[:, 1, :, :][:, None, :, :]
        dxc = offset[:, 2, :, :][:, None, :, :]
        dyc = offset[:, 3, :, :][:, None, :, :]
        dr = offset[:, 4, :, :][:, None, :, :]

        width_transform = scale_transform[0]
        height_transform = scale_transform[1]

        dx1 = dx1 * width_transform / 3
        dy1 = dy1 * height_transform / 3
        dxc = dxc * width_transform / 3
        dyc = dyc * height_transform / 3

        reg = torch.cat((dx1, dy1, dxc, dyc, dr), dim=1)
        return reg

    def forward(self, x):
        cls_logits = []
        bbox_regs = []
        # dcn_offset = {}
        for index, feature in enumerate(x):
            logits = []
            regs = []
            for branch in range(self.num_branch):
                # dcn_offset["branch_{:d}".format(branch)] = []
                # dcn_offset["branch_{:d}".format(branch)].append(offset)
                m = getattr(self, "cls_branch_" + str(branch))

                #                 out, offset, scale_transform = m[0](feature) # [1,32,H,W] [1,5,H,W]
                out = m[0](feature)  # [1,32,H,W] [1,5,H,W]
                #                 reg = offset
                #                 scale_transform_cascade = scale_transform
                out = m[1](out)
                #                 out, offset, scale_transform = m[2](out) # [1,32,H,W] [1,5,H,W]
                out = m[2](out)  # [1,32,H,W] [1,5,H,W]
                # reg = reg + self.feacture_anchor_match(offset, scale_transform_cascade)
                #                 reg = reg + offset
                #                 scale_transform_cascade = scale_transform
                out = m[3](out)

                #                 out, offset, scale_transform = m[4](out) # [1,32,H,W] [1,5,H,W]
                out = m[4](out)  # [1,32,H,W] [1,5,H,W]
                # reg = reg + self.feacture_anchor_match(offset, scale_transform_cascade)
                #                 reg = reg + offset
                #                 scale_transform_cascade = scale_transform
                out = m[5](out)

                out, offset, _ = m[6](out)  # [1,32,H,W] [1,5,H,W]
                # reg = reg + self.feacture_anchor_match(offset, scale_transform_cascade)
                reg = torch.zeros_like(offset)
                reg[:, :4, :, :] = offset[:, :4, :, :] * (2 ** (index + 3))
                reg[:, 4, :, :] = offset[:, 4, :, :]
                # scale_transform_cascade = scale_transform
                out = m[7](out)

                logit = m[8](out)  # [1,15,H,W]
                # anchor size to match

                #                 reg_detch = reg[:, :4, :, :].clone().detach() * (2 ** (index + 3))
                # reg[:, :4, :, :] = reg[:, :4, :, :] * (2 ** (index + 3))
                #                 reg[:, :4, :, :] = reg_detch

                logits.append(logit)
                regs.append(reg)

            cls_logit = logits[0]
            bbox_reg = regs[0]
            for logit, reg in zip(logits[1:], regs[1:]):
                cls_logit = torch.cat((cls_logit, logit), dim=1)
                bbox_reg = torch.cat((bbox_reg, reg), dim=1)

            cls_logits.append(cls_logit)  # [1, 15 * 21, H, W]
            bbox_regs.append(bbox_reg)  # [1, 5 * 21, H, W]

        return cls_logits, bbox_regs


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(RetinaNetModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator_retinanet(cfg)
        if cfg.MODEL.RETINANET_DCN_ON:
            head = DCN_RetinaNetHead(cfg, in_channels)
        else:
            head = RetinaNetHead(cfg, in_channels)
        box_coder = BoxCoder(weights=cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS)

        box_selector_test = make_retinanet_postprocessor(cfg, box_coder, is_train=False)

        loss_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, box_cls, box_regression, targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_train(self, anchors, box_cls, box_regression, targets):

        loss_box_cls, loss_box_reg = self.loss_evaluator(
            anchors, box_cls, box_regression, targets
        )
        losses = {
            "loss_retina_cls": loss_box_cls,
            "loss_retina_reg": loss_box_reg,
        }
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        return boxes, {}


def build_retinanet(cfg, in_channels):
    return RetinaNetModule(cfg, in_channels)
