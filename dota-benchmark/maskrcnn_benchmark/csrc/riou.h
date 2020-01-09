// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


at::Tensor riou(const at::Tensor& ex_dets,
               const at::Tensor& gt_dets) {

  if (ex_dets.type().is_cuda() && gt_dets.type().is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    if (ex_dets.numel() == 0 || gt_dets.numel() == 0)
      return at::empty({0}, ex_dets.options().dtype(at::kLong).device(at::kCPU));
    //auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
    return riou_cuda(ex_dets, gt_dets);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

//  at::Tensor result = nms_cpu(dets, scores, threshold);
//  return result;
}
