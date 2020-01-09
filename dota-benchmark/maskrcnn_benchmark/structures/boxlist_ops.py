# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .bounding_box import BoxList

from maskrcnn_benchmark.layers import nms as _box_nms

from maskrcnn_benchmark.utils.rotate_tools import rotate_utils as trans
from maskrcnn_benchmark.layers import riou
from maskrcnn_benchmark.layers import rnms
from maskrcnn_benchmark.config import cfg


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def boxlist_rnms(boxlist, nms_thresh, max_proposals=-1, score_field="scores", need_keep=False):
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.bbox
    scores = boxlist.get_field(score_field)
    keep = rnms(boxes, scores, nms_thresh)

    boxlist = boxlist[keep]
    if need_keep:
        return boxlist, keep
    return boxlist


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def boxlist_riou(ex_boxlist, gt_boxlist):
    """计算xywhd五点数据的iou
    Arguments:
      box1: (BoxList) bounding boxes, sized [N,5].
      box2: (BoxList) bounding boxes, sized [M,5].

    Returns:
      (tensor) iou, sized [M,N].
    """
    if ex_boxlist.size != gt_boxlist.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(ex_boxlist, gt_boxlist))
    ex_len = len(ex_boxlist)
    gt_len = len(gt_boxlist)

    ex_boxes, gt_boxes = ex_boxlist.bbox, gt_boxlist.bbox

    if ex_boxes.size(1) == 4 and gt_boxes.size(1) == 8:  # first stage anchor match target, rect match rect
        ex_xmin, ex_ymin, ex_xmax, ex_ymax = ex_boxes.split(1, dim=-1)
        ex_boxes = torch.cat((ex_xmin, ex_ymin, ex_xmax, ex_ymin,
                              ex_xmax, ex_ymax, ex_xmin, ex_ymax), dim=1)

        x1, y1, x2, y2, x3, y3, x4, y4 = gt_boxes.split(1, dim=-1)
        gt_x = torch.cat((x1, x2, x3, x4), dim=1)
        gt_y = torch.cat((y1, y2, y3, y4), dim=1)

        gt_xmin, _ = gt_x.min(1)
        gt_xmax, _ = gt_x.max(1)
        gt_ymin, _ = gt_y.min(1)
        gt_ymax, _ = gt_y.max(1)

        gt_boxes = torch.cat((gt_xmin[:, None], gt_ymin[:, None], gt_xmax[:, None], gt_ymin[:, None],
                              gt_xmax[:, None], gt_ymax[:, None], gt_xmin[:, None], gt_ymax[:, None]), dim=1)

        iou = riou(ex_boxes, gt_boxes)
        iou = iou.reshape(ex_len, gt_len).transpose(0, 1)

        return iou

    elif ex_boxes.size(1) == 8 and gt_boxes.size(1) == 8:  # second stage proposal match target, poly match poly
        if cfg.R2CNN:
            x1, y1, x2, y2, x3, y3, x4, y4 = gt_boxes.split(1, dim=-1)
            gt_x = torch.cat((x1, x2, x3, x4), dim=1)
            gt_y = torch.cat((y1, y2, y3, y4), dim=1)

            gt_xmin, _ = gt_x.min(1)
            gt_xmax, _ = gt_x.max(1)
            gt_ymin, _ = gt_y.min(1)
            gt_ymax, _ = gt_y.max(1)

            gt_boxes = torch.cat((gt_xmin[:, None], gt_ymin[:, None], gt_xmax[:, None], gt_ymin[:, None],
                                  gt_xmax[:, None], gt_ymax[:, None], gt_xmin[:, None], gt_ymax[:, None]), dim=1)
            iou = riou(ex_boxes, gt_boxes)
            iou = iou.reshape(ex_len, gt_len).transpose(0, 1)

        else:
            iou = riou(ex_boxes, gt_boxes)
            iou = iou.reshape(ex_len, gt_len).transpose(0, 1)

        return iou


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
