# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
helper class that supports empty tensors on some nn functions.

Ideally, add support directly in PyTorch to empty tensors in
those functions.

This can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

import math
import torch
from torch import nn
from torch.nn.modules.utils import _ntuple

from maskrcnn_benchmark.config import cfg


def rotate2deformable(offset, scale_transform=False):
    dx1 = offset[:, 0, :, :][:, None, :, :]
    dy1 = offset[:, 1, :, :][:, None, :, :]
    dxc = offset[:, 2, :, :][:, None, :, :]
    dyc = offset[:, 3, :, :][:, None, :, :]
    dr = offset[:, 4, :, :][:, None, :, :]

    x01 = dx1
    y01 = dy1 + 1
    x11 = 1 + dxc
    y11 = 1 + dyc

    dist_k11_k01 = torch.sqrt(
        (y01 - y11) ** 2 + (x01 - y11) ** 2
    )
    dist_k01_k00 = dist_k11_k01 * torch.exp(dr)

    sint = torch.abs(y01 - y11) / dist_k11_k01
    cost = torch.abs(x01 - x11) / dist_k11_k01

    x00 = x01 - dist_k01_k00 * sint
    y00 = y01 - dist_k01_k00 * cost
    dx0 = x00 - 0
    dy0 = y00 - 0

    x02 = x01 * 2 - x00
    y02 = y01 * 2 - y00
    dx2 = x02 - 0
    dy2 = y02 - 2

    x10 = x11 - dist_k01_k00 * sint
    y10 = y11 - dist_k01_k00 * cost
    dx3 = x10 - 1
    dy3 = y10 - 0

    x12 = x11 * 2 - x10
    y12 = y11 * 2 - y10
    dx5 = x12 - 1
    dy5 = y12 - 2

    x20 = x10 * 2 - x00
    y20 = y10 * 2 - y00
    dx6 = x20 - 2
    dy6 = y20 - 0

    x21 = x11 * 2 - x01
    y21 = y11 * 2 - y01
    dx7 = x21 - 2
    dy7 = y21 - 1

    x22 = x11 * 2 - x00
    y22 = y11 * 2 - y00
    dx8 = x22 - 2
    dy8 = y22 - 2

    offset_18 = torch.cat((dx0, dy0, dx1, dy1, dx2, dy2,
                           dx3, dy3, dxc, dyc, dx5, dy5,
                           dx6, dy6, dx7, dy7, dx8, dy8), dim=1)
    if scale_transform:
        width = torch.sqrt((x02 - x00) ** 2 + (y02 - y00) ** 2) + 1
        height = torch.sqrt((x20 - x00) ** 2 + (y20 - y00) ** 2) + 1
        return offset_18, [width, height]
    return offset_18


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        # get output shape

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ConvTranspose2d(torch.nn.ConvTranspose2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).forward(x)
        # get output shape

        output_shape = [
            (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class BatchNorm2d(torch.nn.BatchNorm2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm2d, self).forward(x)
        # get output shape
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if (
            scale_factor is not None
            and isinstance(scale_factor, tuple)
            and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
        ]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)


class DFConv2d(nn.Module):
    """Deformable convolutional layer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        with_modulated_dcn=True,
        kernel_size=3,
        stride=1,
        groups=1,
        dilation=1,
        deformable_groups=1,
        bias=False,
        need_offset=False
    ):
        self.need_offset = need_offset

        super(DFConv2d, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert isinstance(stride, (list, tuple))
            assert isinstance(dilation, (list, tuple))
            assert len(kernel_size) == 2
            assert len(stride) == 2
            assert len(dilation) == 2
            padding = (
                dilation[0] * (kernel_size[0] - 1) // 2,
                dilation[1] * (kernel_size[1] - 1) // 2
            )
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            padding = dilation * (kernel_size - 1) // 2
            offset_base_channels = kernel_size * kernel_size
        if with_modulated_dcn:
            from maskrcnn_benchmark.layers import ModulatedDeformConv
            # offset_channels = offset_base_channels * 3 #default: 27
            offset_channels = cfg.MODEL.DCN.OFFSET
            conv_block = ModulatedDeformConv
        else:
            from maskrcnn_benchmark.layers import DeformConv
            # offset_channels = offset_base_channels * 2 #default: 18
            offset_channels = cfg.MODEL.DCN.OFFSET
            conv_block = DeformConv
        self.offset = Conv2d(
            in_channels,
            deformable_groups * offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            dilation=dilation
        )
        for l in [self.offset, ]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            torch.nn.init.constant_(l.bias, 0.)
        self.conv = conv_block(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deformable_groups=deformable_groups,
            bias=bias
        )
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset = self.offset(x)
                if cfg.MODEL.RETINANET_DCN_ON:
                    offset_18, scale_transform = rotate2deformable(offset, scale_transform=True)
                    x = self.conv(x, offset_18)
                    return x, offset, scale_transform
                else:
                    x = self.conv(x, offset)
            else:
                offset_mask = self.offset(x)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            if self.need_offset:
                return x, offset
            return x
        # get output shape
        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride
            )
        ]
        output_shape = [x.shape[0], self.conv.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)
