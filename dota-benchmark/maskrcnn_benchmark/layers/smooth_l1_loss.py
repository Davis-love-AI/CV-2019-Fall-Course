# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True, iou=None, multi_reg=False):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta

    if multi_reg:
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        #         return loss.sum(1)[:, None]
        return loss

    elif iou is None:
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if size_average:
            return loss.mean()
        return loss.sum()

    else:
        iou_factor = torch.abs(- torch.log(iou.clone().detach()))
        _loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        _loss_ = _loss.sum(1)[:, None].detach()
        loss = _loss / _loss_ * iou_factor

        if size_average:
            return loss.mean()
        return loss.sum()
