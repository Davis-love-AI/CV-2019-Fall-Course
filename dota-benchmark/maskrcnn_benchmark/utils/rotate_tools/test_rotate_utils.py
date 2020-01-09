import torch

from maskrcnn_benchmark.layers import rnms as _box_rnms
from maskrcnn_benchmark.layers import nms as _box_nms
from maskrcnn_benchmark.layers import riou as _riou
from maskrcnn_benchmark.utils import rotate_utils as trans
import numpy as np
import cv2


boxes = torch.tensor(
    [
        [10., 10., 10., 20., 20., 20., 20., 10.],
        [25, 15, 15, 15, 15, 25, 25, 25],
        [12, 12, 12, 21, 21, 21, 21, 12],
        [5, 5, 5, 14, 14, 14, 14, 5]
    ]
)

scores = torch.tensor(
    [
        0.82123,
        0.8123,
        0.91,
        0.81232
    ]
)
keep = trans.inclined_nms(trans.xy2wh(boxes).numpy(), scores.numpy(), 0.4, 2000)
print(keep)
keep = _box_rnms(boxes.cuda(), scores.cuda(), 0.4)
print(keep)

ex_boxes = torch.tensor(
    [
        [10., 10., 10., 20., 20., 20., 20., 10.],
        [25, 15, 15, 15, 15, 25, 25, 25],
        [17, 17, 17, 6, 6, 6, 6, 17]
    ], dtype=torch.float32
)
gt_boxes = torch.tensor(
    [
        [12, 12, 12, 21, 21, 21, 21, 12],
        [14, 14, 14, 5, 5, 5, 5, 14, ]
    ], dtype=torch.float32
)

iou = trans.inclined_iou(trans.xy2wh(ex_boxes.numpy()), trans.xy2wh(gt_boxes.numpy()))
print(iou)
overlap = _riou(ex_boxes.cuda(), gt_boxes.cuda()).reshape(len(ex_boxes), len(gt_boxes)).transpose(0, 1)
print(overlap)

# boxes = [[0, 0, 0, 4, 5, 4, 5, 0], [0, 0, 0, 10, 10, 10, 10, 0]]
image = 255 * np.ones((512, 512), dtype=np.uint8)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

boxes = np.array([[100, 100, 200, 230]])
# cv2.rectangle(image, tuple(boxes[0][:2]), tuple(boxes[0][2:]), (123))

boxes_r = trans.rotated_anchors(boxes, [60, 150]).numpy()
for box in boxes_r:
    cv2.polylines(image, np.array(box.reshape(1, 4, 2), dtype=np.int32), True, (230))

boxes5pi = trans.encode(boxes_r).numpy()
boxes_conv4 = trans.convert4(boxes5pi).cpu().numpy()
# cv2.rectangle(image, tuple(boxes_conv4[0][:2]), tuple(boxes_conv4[0][2:]), (190))

boxes_conv8 = trans.convert8(boxes5pi).cpu().numpy()
for box in boxes_conv8:
    cv2.polylines(image, np.array(box.reshape(1, 4, 2), dtype=np.int32), True, (10))
cv2.imshow('image', image)
cv2.waitKey(0)

# box = wh2xy(decode(boxes5pi))
# boxes5a = decode(boxes5pi)
# boxesxy = wh2xy(boxes5a)
print('xyxy 4', boxes)
print('(xyxyxyxy)8:', boxes_r)
print('(xywha(pi))5', boxes5pi)
print('(xyxy)4 conv4', boxes_conv4)
# print('(xywha(cv))5', boxes5a)
# print('(xyxyxyxy)8 from decode:', boxesxy)
# print('box', box)
# boxes = quad2inrect(boxes_r)
# print('mode=xywht\n', boxes)

# boxes = wh2xy(boxes)
# print('mode=xyxyxyxy\n', boxes)

# boxes = xy2wh(boxes)
# print(boxes)
