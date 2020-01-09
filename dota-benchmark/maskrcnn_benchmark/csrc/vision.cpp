// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "SigmoidFocalLoss.h"

#include "ActiveRotatingFilter.h"
#include "RotationInvariantEncoding.h"

#include "riou.h"
#include "rnms.h"


#include "deform_conv.h"
#include "deform_pool.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
  m.def("sigmoid_focalloss_forward", &SigmoidFocalLoss_forward, "SigmoidFocalLoss_forward");
  m.def("sigmoid_focalloss_backward", &SigmoidFocalLoss_backward, "SigmoidFocalLoss_backward");
    
  m.def("arf_mappingrotate_forward", &ARF_MappingRotate_forward, "active rotating filter forward");
  m.def("arf_mappingrotate_backward", &ARF_MappingRotate_backward, "active rotating filter backward");
  m.def("rie_alignfeature_forward", &RIE_AlignFeature_forward, "rotation invariant encoding forward");
  m.def("rie_alignfeature_backward", &RIE_AlignFeature_backward, "rotation invariant encoding backward");

  m.def("rnms", &rnms, "inclind non-maximum suppression");
  m.def("riou", &riou, "inclind Intersection-over-Union");

  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
  m.def("deform_conv_backward_input", &deform_conv_backward_input, "deform_conv_backward_input");
  m.def("deform_conv_backward_parameters", &deform_conv_backward_parameters, "deform_conv_backward_parameters");
  m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward, "modulated_deform_conv_forward");
  m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward, "modulated_deform_conv_backward");
  m.def("deform_psroi_pooling_forward", &deform_psroi_pooling_forward, "deform_psroi_pooling_forward");
  m.def("deform_psroi_pooling_backward", &deform_psroi_pooling_backward, "deform_psroi_pooling_backward");

}
