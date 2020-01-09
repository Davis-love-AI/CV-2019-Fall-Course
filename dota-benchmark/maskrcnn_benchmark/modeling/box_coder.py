# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils import rotate_utils as trans


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        if not cfg.ROTATE:
            TO_REMOVE = 1  # TODO remove
            ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
            ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
            ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
            ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

            gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
            gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
            gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
            gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

            wx, wy, ww, wh = self.weights
            targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
            targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
            targets_dw = ww * torch.log(gt_widths / ex_widths)
            targets_dh = wh * torch.log(gt_heights / ex_heights)

            targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

            return targets

        elif cfg.ROTATE and not "RETINANET" in cfg.MODEL.BACKBONE.CONV_BODY:
            TO_REMOVE = 1  # TODO remove
            ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
            ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
            ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
            ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

            gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
            gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
            gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
            gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

            wx, wy, ww, wh = self.weights
            targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
            targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
            targets_dw = ww * torch.log(gt_widths / ex_widths)
            targets_dh = wh * torch.log(gt_heights / ex_heights)

            targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

            return targets

        elif cfg.ROTATE and "RETINANET" in cfg.MODEL.BACKBONE.CONV_BODY:
            if cfg.MODEL.RETINANET_DCN_ON:
                TO_REMOVE = 1

                ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
                ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE

                ex_top_x = proposals[:, 0] + 0.5 * ex_widths
                ex_top_y = proposals[:, 1]
                ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
                ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights
                ex_rescale = torch.log(ex_widths / ex_heights)

                gt_top_x = reference_boxes[:, 0]
                gt_top_y = reference_boxes[:, 1]
                gt_ctr_x = reference_boxes[:, 2]
                gt_ctr_y = reference_boxes[:, 3]
                gt_rescale = reference_boxes[:, 4]

                wx1, wy1, wxc, wyc, wr = self.weights
                targets_dx1 = wx1 * (gt_top_x - ex_top_x) / ex_widths
                targets_dy1 = wy1 * (gt_top_y - ex_top_y) / ex_heights
                targets_dxc = wxc * (gt_ctr_x - ex_ctr_x) / ex_widths
                targets_dyc = wyc * (gt_ctr_y - ex_ctr_y) / ex_heights
                targets_dr = wr * (gt_rescale - ex_rescale)  # / 3.1415926
                targets = torch.stack((targets_dx1, targets_dy1, targets_dxc, targets_dyc, targets_dr), dim=1)

                return targets
            else:
                TO_REMOVE = 1

                ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
                ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
                ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
                ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights
                ex_theta = torch.ones_like(ex_widths) * (- 3.14 / 2)

                gt_ctr_x = reference_boxes[:, 0]
                gt_ctr_y = reference_boxes[:, 1]
                gt_widths = reference_boxes[:, 2]
                gt_heights = reference_boxes[:, 3]
                gt_theta = reference_boxes[:, 4]

                wx, wy, ww, wh, wt = self.weights
                targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
                targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
                targets_dw = ww * torch.log(gt_widths / ex_widths)
                targets_dh = wh * torch.log(gt_heights / ex_heights)
                targets_dt = wt * (gt_theta - ex_theta)  # / 3.1415926
                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, targets_dt), dim=1)

                return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        boxes = boxes.to(rel_codes.dtype)
        if "RETINANET" in cfg.MODEL.BACKBONE.CONV_BODY:
            if cfg.MODEL.RETINANET_DCN_ON:
                TO_REMOVE = 1  # TODO remove
                widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
                heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
                top_x = boxes[:, 0]
                top_y = boxes[:, 1]
                ctr_x = boxes[:, 0] + 0.5 * widths
                ctr_y = boxes[:, 1] + 0.5 * heights
                rescale = torch.log(widths / heights)

                wx1, wy1, wxc, wyc, wr = self.weights
                dx1 = rel_codes[:, 0::5] / wx1
                dy1 = rel_codes[:, 1::5] / wy1
                dxc = rel_codes[:, 2::5] / wxc
                dyc = rel_codes[:, 3::5] / wyc
                dr = rel_codes[:, 4::5] / wr

                pred_top_x = dx1 * widths[:, None] + top_x[:, None]
                pred_top_y = dy1 * heights[:, None] + top_y[:, None]
                pred_ctr_x = dxc * widths[:, None] + ctr_x[:, None]
                pred_ctr_y = dyc * heights[:, None] + ctr_y[:, None]
                pred_r = dr + rescale[:, None]  # dt * 3.1415926 + theta[:, None]

                pred_boxes = torch.zeros_like(rel_codes)

                pred_boxes[:, 0::5] = pred_top_x
                pred_boxes[:, 1::5] = pred_top_y
                pred_boxes[:, 2::5] = pred_ctr_x
                pred_boxes[:, 3::5] = pred_ctr_y
                pred_boxes[:, 4::5] = pred_r

                return pred_boxes
            else:
                TO_REMOVE = 1  # TODO remove
                widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
                heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
                ctr_x = boxes[:, 0] + 0.5 * widths
                ctr_y = boxes[:, 1] + 0.5 * heights
                theta = torch.ones_like(widths) * (- 3.14 / 2)

                wx, wy, ww, wh, wt = self.weights
                dx = rel_codes[:, 0::5] / wx
                dy = rel_codes[:, 1::5] / wy
                dw = rel_codes[:, 2::5] / ww
                dh = rel_codes[:, 3::5] / wh
                dt = rel_codes[:, 4::5] / wt

                # Prevent sending too large values into torch.exp()
                dw = torch.clamp(dw, max=self.bbox_xform_clip)
                dh = torch.clamp(dh, max=self.bbox_xform_clip)

                pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
                pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
                pred_w = torch.exp(dw) * widths[:, None]
                pred_h = torch.exp(dh) * heights[:, None]
                pred_t = dt + theta[:, None]  # dt * 3.1415926 + theta[:, None]

                pred_boxes = torch.zeros_like(rel_codes)

                pred_boxes[:, 0::5] = pred_ctr_x
                pred_boxes[:, 1::5] = pred_ctr_y
                pred_boxes[:, 2::5] = pred_w
                pred_boxes[:, 3::5] = pred_h
                pred_boxes[:, 4::5] = pred_t

                return pred_boxes

        else:
            TO_REMOVE = 1  # TODO remove
            widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
            heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
            ctr_x = boxes[:, 0] + 0.5 * widths
            ctr_y = boxes[:, 1] + 0.5 * heights

            wx, wy, ww, wh = self.weights
            dx = rel_codes[:, 0::4] / wx
            dy = rel_codes[:, 1::4] / wy
            dw = rel_codes[:, 2::4] / ww
            dh = rel_codes[:, 3::4] / wh

            # Prevent sending too large values into torch.exp()
            dw = torch.clamp(dw, max=self.bbox_xform_clip)
            dh = torch.clamp(dh, max=self.bbox_xform_clip)

            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
            pred_w = torch.exp(dw) * widths[:, None]
            pred_h = torch.exp(dh) * heights[:, None]

            pred_boxes = torch.zeros_like(rel_codes)
            # x1
            pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
            # y1
            pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
            # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
            pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
            # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
            pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

            return pred_boxes


class BoxCoder2(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        #         if len(self.weights) == 4:
        if reference_boxes.size(1) == 4:
            TO_REMOVE = 1  # TODO remove
            ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
            ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
            ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
            ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

            gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
            gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
            gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
            gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

            wx, wy, ww, wh, _ = self.weights
            targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
            targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
            targets_dw = ww * torch.log(gt_widths / ex_widths)
            targets_dh = wh * torch.log(gt_heights / ex_heights)

            targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

            return targets

        elif reference_boxes.size(1) == 5:
            # TO_REMOVE = 1  # TODO remove
            # ex_xmin, ex_ymin, ex_xmax, ex_ymax = proposals.split(1, dim=-1)
            # proposals = torch.cat((ex_xmin, ex_ymin, ex_xmax, ex_ymin,
            #                         ex_xmax, ex_ymax, ex_xmin, ex_ymax), dim=1)
            # _proposals = trans.xy2wh_tensor(proposals)
            # _reference_boxes = trans.xy2wh_tensor(reference_boxes)

            ex_ctr_x = proposals[:, 0]
            ex_ctr_y = proposals[:, 1]
            ex_widths = proposals[:, 2]
            ex_heights = proposals[:, 3]
            ex_theta = proposals[:, 4]

            gt_ctr_x = reference_boxes[:, 0]
            gt_ctr_y = reference_boxes[:, 1]
            gt_widths = reference_boxes[:, 2]
            gt_heights = reference_boxes[:, 3]
            gt_theta = reference_boxes[:, 4]

            wx, wy, ww, wh, wt = self.weights
            targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
            targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
            targets_dw = ww * torch.log(gt_widths / ex_widths)
            targets_dh = wh * torch.log(gt_heights / ex_heights)
            targets_dt = wt * (gt_theta - ex_theta)  # / 3.1415926
            targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, targets_dt), dim=1)
            return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        boxes = boxes.to(rel_codes.dtype)
        if boxes.size(1) == 4:
            TO_REMOVE = 1  # TODO remove
            widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
            heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
            ctr_x = boxes[:, 0] + 0.5 * widths
            ctr_y = boxes[:, 1] + 0.5 * heights

            wx, wy, ww, wh, _ = self.weights
            dx = rel_codes[:, 0::4] / wx
            dy = rel_codes[:, 1::4] / wy
            dw = rel_codes[:, 2::4] / ww
            dh = rel_codes[:, 3::4] / wh

            # Prevent sending too large values into torch.exp()
            dw = torch.clamp(dw, max=self.bbox_xform_clip)
            dh = torch.clamp(dh, max=self.bbox_xform_clip)

            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
            pred_w = torch.exp(dw) * widths[:, None]
            pred_h = torch.exp(dh) * heights[:, None]

            pred_boxes = torch.zeros_like(rel_codes)
            # x1
            pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
            # y1
            pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
            # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
            pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
            # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
            pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

            return pred_boxes

        if boxes.size(1) == 5:

            widths = boxes[:, 2]
            heights = boxes[:, 3]
            ctr_x = boxes[:, 0]
            ctr_y = boxes[:, 1]
            theta = boxes[:, 4]

            wx, wy, ww, wh, wt = self.weights
            dx = rel_codes[:, 0::5] / wx
            dy = rel_codes[:, 1::5] / wy
            dw = rel_codes[:, 2::5] / ww
            dh = rel_codes[:, 3::5] / wh
            dt = rel_codes[:, 4::5] / wt

            # Prevent sending too large values into torch.exp()
            dw = torch.clamp(dw, max=self.bbox_xform_clip)
            dh = torch.clamp(dh, max=self.bbox_xform_clip)

            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
            pred_w = torch.exp(dw) * widths[:, None]
            pred_h = torch.exp(dh) * heights[:, None]
            pred_t = dt + theta[:, None]  # dt * 3.1415926 + theta[:, None]

            pred_boxes = torch.zeros_like(rel_codes)

            pred_boxes[:, 0::5] = pred_ctr_x
            pred_boxes[:, 1::5] = pred_ctr_y
            pred_boxes[:, 2::5] = pred_w
            pred_boxes[:, 3::5] = pred_h
            pred_boxes[:, 4::5] = pred_t

            return pred_boxes
