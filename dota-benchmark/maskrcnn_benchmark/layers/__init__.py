# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .batch_norm import FrozenBatchNorm2d
from .misc import Conv2d
from .misc import ConvTranspose2d
from .misc import BatchNorm2d
from .misc import interpolate
from .nms import nms
from .roi_align import ROIAlign
from .roi_align import roi_align
from .roi_pool import ROIPool
from .roi_pool import roi_pool
from .smooth_l1_loss import smooth_l1_loss
from .sigmoid_focal_loss import SigmoidFocalLoss

from .ARF import MappingRotate
from .RIE import ORAlign2d
from .rnms import rnms
from .riou import riou

from .dcn.deform_conv_func import deform_conv, modulated_deform_conv
from .dcn.deform_conv_module import DeformConv, ModulatedDeformConv, ModulatedDeformConvPack
from .dcn.deform_pool_func import deform_roi_pooling
from .dcn.deform_pool_module import DeformRoIPooling, DeformRoIPoolingPack, ModulatedDeformRoIPoolingPack
from .misc import DFConv2d


def mapping_rotate(input, indices):
    return MappingRotate()(input, indices)


def oraligned2d(input, nOrientation):
    return ORAlign2d()(input, nOrientation)


__all__ = ["nms", "roi_align", "ROIAlign", "roi_pool", "ROIPool",
           "smooth_l1_loss", "Conv2d", "ConvTranspose2d", "interpolate",
           "BatchNorm2d", "FrozenBatchNorm2d", "SigmoidFocalLoss",
           "DFConv2d", "oraligned2d", "mapping_rotate", "riou", "rnms",
           'deform_conv', 'modulated_deform_conv', 'DeformConv', 'ModulatedDeformConv',
           'ModulatedDeformConvPack', 'deform_roi_pooling', 'DeformRoIPooling',
           'DeformRoIPoolingPack', 'ModulatedDeformRoIPoolingPack']
