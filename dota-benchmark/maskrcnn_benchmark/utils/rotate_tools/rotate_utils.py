import cv2
import numpy as np
import torch


TO_REMOVE = 1


def quad2inrect(coordinate):
    """ 将原始的dota数据的多边形坐标(8点)坐标修正为斜矩形坐标(8点)
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4] is quadrangle
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4] is inclined rectangle
                                                    but the sequence is not the same
            dtype numpy.ndarray or torch.tensor
    """
    if not isinstance(coordinate, np.ndarray):
        np.array(coordinate, dtype=np.float32)

    boxes_poly5 = []
    boxes_poly8 = []
    for rect in coordinate:
        box = np.int0(rect)
        box = box.reshape([4, 2])
        rect1 = cv2.minAreaRect(box)

        x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
        boxes_poly5.append([x, y, w, h, theta])

    for rect in boxes_poly5:
        box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
        box = np.reshape(box, [-1, ])
        boxes_poly8.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

    # return np.array(boxes_poly8, dtype=np.float32)
    return torch.as_tensor(boxes_poly8)


def encode(coordinate):
    """ 将8坐标点数据转换为5点数据,并且将角度改为弧度制,范围为 (-1/2 pi ~ 0 pi)
        长边为width,短边为height, 形成标注进行回归(用于anchor和gt_boxes)
        :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4]
        :return: format format [x_c, y_c, w(axis_x),
                        h(axis_y),theta(-1/2 pi ~ 0 pi)]
        """
    if not isinstance(coordinate, np.ndarray):
        np.array(coordinate, dtype=np.float32)

    boxes = []
    for rect in coordinate:
        box = np.int0(rect)
        box = box.reshape([4, 2])
        rect1 = cv2.minAreaRect(box)

        x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
        ## 把cv2的格式转换为数学格式
        #        w += TO_REMOVE
        #        h += TO_REMOVE
        theta = theta / 180 * np.pi
        # if w < h:
        #     temp = h
        #     h = w
        #     w = temp
        #     theta = (theta + 90) / 180 * np.pi
        # elif w > h:
        #     theta = theta / 180 * np.pi
        # elif w == h:
        #     w = w + 1e-2
        #     theta = theta / 180 * np.pi

        boxes.append([x, y, w, h, theta])
    # return np.array(boxes, dtype=np.float32)
    return torch.as_tensor(boxes)


def decode(coordinate):
    """ 将encode的5坐标形式下的width, height, angle转换为opencv下的数据形式,用于IoU,NMS计算
    :param coordinate: format [x_c, y_c, w(axis_x),
                        h(axis_y),theta(-1/2 pi ~ 0 pi)]
    :return: format [x_c, y_c, w(x first touch(x坐标轴逆时针旋转碰到的边)),
                        h(另一个边),theta(-90, 0: cv2.dtype)]
    """
    if not isinstance(coordinate, np.ndarray):
        np.array(coordinate, dtype=np.float32)

    boxes = []
    for rect in coordinate:
        x_ctr, y_ctr, w, h, theta = rect

        w -= TO_REMOVE
        h -= TO_REMOVE
        theta = theta / np.pi * 180

        if theta >= 0:  ## 角度应该减去90度(encode是加上90度) w, h互换
            temp = h
            h = w
            w = temp
            theta -= 90
        else:
            theta = theta

        boxes.append([x_ctr, y_ctr, w, h, theta])

    # return np.array(boxes, dtype=np.float32)
    return torch.as_tensor(boxes)


def convert4(coordinate, mode='xyxy'):
    """ 将5点数据转换为4点数据,用于判断anchor 或 gt 或 proposal 是否越界(图片边界)
    :param cooronate: format [x_c, y_c, w(long_side),
                        h(short_side),theta(-1/2 pi ~ 1/2 pi)]
    :return: poly4(xyxy) format [xmin, ymin, xmax, ymax] 这是4个点不是为了确定一个水平矩形框
             poly4(xywh) format [x_ctr, y_ctr, w(long_side), h(short_side)]
    """
    if not isinstance(coordinate, torch.Tensor):
        coordinate = torch.FloatTensor(coordinate).cuda()

    # for box in bbox:
    x, y, w, h, d = coordinate.split(1, dim=-1)

    d = torch.abs(d)
    sind = torch.sin(d)
    cosd = torch.cos(d)
    dw = w / 2
    dh = h / 2

    xmin = x - (dw * cosd + dh * sind)
    ymin = y - (dw * sind + dh * cosd)
    xmax = 2 * x - xmin
    ymax = 2 * y - ymin

    if mode == 'xyxy':  # 由5tuple xywht 转换为自然目标检测的xyxy
        poly = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
    elif mode == 'xywht':  # 由5tuple xywht 转换为遥感检测的xywh(其中w>h, xy为中心值,t为0,在函数外部考虑)
        x_ctr = (xmin + xmax) / 2
        y_ctr = (ymin + ymax) / 2
        w = (xmax - xmin)
        h = (ymax - ymin)
        # opencv 格式下的w和h是与正常的相反的 所有w h 应该互换， 角度是-1.57
        poly = torch.cat((x_ctr, y_ctr, h, w), dim=-1)

    elif mode == 'xywh':  # 由5tuple xywht 转换为遥感目标检测的xywh(xy为中心值ps:自然目标检测xy是左上角值)
        x_ctr = (xmin + xmax) / 2
        y_ctr = (ymin + ymax) / 2
        w = (xmax - xmin)
        h = (ymax - ymin)
        poly = torch.cat((x_ctr, y_ctr, w, h), dim=-1)
    else:
        raise ValueError("mode should be 'xyxy' or 'xywh' or 'xywht'")

    return poly


def convert8(coordinate):
    """ 将5点数据转换为8点数据,用于判断用于inclined iou 和inclined nms
    :param cooronate: format [x_c, y_c, w(long_side),
                        h(short_side),theta(-1/2 pi ~ 1/2 pi)]
    :return: poly8 format [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    if not isinstance(coordinate, torch.Tensor):
        coordinate = torch.as_tensor(coordinate).cuda()
    # for box in bbox:
    x, y, w, h, d = coordinate.split(1, dim=-1)

    _d = torch.abs(d)
    sind = torch.sin(_d)
    cosd = torch.cos(_d)
    dw = w / 2
    dh = h / 2

    # d_cpu = d.cpu().numpy()
    # _0 = np.where(d_cpu <= 0)
    # __0 = np.where(d_cpu > 0)

    y1 = torch.zeros_like(y)
    x2 = torch.zeros_like(x)

    x1 = x - dw * cosd - dh * sind
    y1 = y + dw * sind - dh * cosd
    # y1[_0] = y[_0] + dw[_0] * sind[_0] - dh[_0] * cosd[_0]
    # y1[__0] = y[__0] - dw[__0] * sind[__0] + dh[__0] * cosd[__0]

    # x2[__0] = x[__0] - dw[__0] * cosd[__0] + dh[__0] * sind[__0]
    # x2[_0] = x[_0] + dw[_0] * cosd[_0] - dh[_0] * sind[_0]
    x2 = x + dw * cosd - dh * sind
    y2 = y - dw * sind - dh * cosd

    x3 = 2 * x - x1
    y3 = 2 * y - y1

    x4 = 2 * x - x2
    y4 = 2 * y - y2

    poly = torch.cat((x1, y1, x2, y2, x3, y3, x4, y4), dim=-1)

    return poly


def xy2wh(coordinate):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: format [x_c, y_c, w, h, theta]
    """
    if isinstance(coordinate, torch.Tensor):
        coordinate = coordinate.numpy()
    if not isinstance(coordinate, np.ndarray):
        np.array(coordinate, dtype=np.float32)

    boxes = []
    for rect in coordinate:
        box = np.int0(rect)
        box = box.reshape([4, 2])
        rect1 = cv2.minAreaRect(box)

        x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
        boxes.append([x, y, w, h, theta])

    return torch.as_tensor(boxes)
    # return np.array(boxes, dtype=np.float32)


def xy2wh_tensor(coordinate):
    k1 = (coordinate[:, 3] - coordinate[:, 1]) / (coordinate[:, 2] - coordinate[:, 0])
    k2 = (coordinate[:, 7] - coordinate[:, 1]) / (coordinate[:, 6] - coordinate[:, 0])

    _w = (
             (coordinate[:, 3] - coordinate[:, 1]) ** 2 +
             (coordinate[:, 2] - coordinate[:, 0]) ** 2
         ) ** 0.5
    _h = (
             (coordinate[:, 7] - coordinate[:, 1]) ** 2 +
             (coordinate[:, 6] - coordinate[:, 0]) ** 2
         ) ** 0.5

    theta1 = torch.atan(k1) / np.pi * 180
    theta2 = torch.atan(k2) / np.pi * 180
    theta = torch.where(theta1 < theta2, theta1, theta2)

    w = torch.where(theta1 < theta2, _w, _h)
    h = torch.where(theta1 < theta2, _h, _w)
    x_ctr = (coordinate[:, 4] + coordinate[:, 0]) / 2
    y_ctr = (coordinate[:, 5] + coordinate[:, 1]) / 2
    poly5 = torch.cat((x_ctr[:, None], y_ctr[:, None], w[:, None], h[:, None], theta[:, None]), dim=1)

    return poly5


def wh2xy(coordinate):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    if isinstance(coordinate, torch.Tensor):
        coordinate = coordinate.numpy()
    if not isinstance(coordinate, np.ndarray):
        np.array(coordinate, dtype=np.float32)

    boxes = []
    for rect in coordinate:
        _box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
        # box = np.reshape(box, [-1, ])
        box = [_box[0][0], _box[0][1], _box[1][0], _box[1][1],
               _box[2][0], _box[2][1], _box[3][0], _box[3][1]]
        boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])
    _boxes = np.array(boxes)
    return torch.as_tensor(_boxes)


def rotated_anchors(rect_anchors, degree):
    """
    :param rect_anchors:  format [[xmin, ymin, xmax, ymax], [xmin, ymin ...]...] #生成的anchor数组
    :param degree: format [degree1, degree2, ...] #需要旋转的角度
    :return:
    """
    if not isinstance(rect_anchors, np.ndarray):
        np.array(rect_anchors, dtype=np.float32)

    x1s = rect_anchors[:, 0]
    y1s = rect_anchors[:, 1]
    x3s = rect_anchors[:, 2]
    y3s = rect_anchors[:, 3]
    x2s = x1s
    y2s = y3s
    x4s = x3s
    y4s = y1s

    widths = x3s - x1s + 1
    heights = y3s - y1s + 1
    x_ctrs = x1s + 0.5 * (widths - 1)
    y_ctrs = y1s + 0.5 * (heights - 1)

    # get the rotated matrix
    rotated_matrix = []
    boxes_rotated = []
    anchors = []
    # degree = (0 - np.array(degree)).tolist()
    for x_ctr, y_ctr in zip(x_ctrs, y_ctrs):
        for rotated_angle in degree:
            rotated_matrix_temp = cv2.getRotationMatrix2D((x_ctr, y_ctr), rotated_angle, 1)
            rotated_matrix.append(rotated_matrix_temp)
    rotated_matrix = np.array(rotated_matrix).reshape(-1, len(degree), 2, 3)  # (x anchors, y degrees, 2, 3)

    # rotate the anchors
    for index, (x1, y1, x2, y2, x3, y3, x4, y4) in enumerate(zip(x1s, y1s, x2s, y2s, x3s, y3s, x4s, y4s)):
        box_matrix = np.vstack((np.array([x1, y1, 1]), np.array([x2, y2, 1]),
                                np.array([x3, y3, 1]), np.array([x4, y4, 1]))).T

        box_rotated = np.dot(rotated_matrix[index], box_matrix)
        boxes_rotated.append(box_rotated)
    boxes_rotated = np.array(boxes_rotated).reshape(-1, 8)

    # add to anchors
    for boxes in boxes_rotated:
        anchors.append([boxes[0], boxes[4], boxes[1], boxes[5],
                        boxes[2], boxes[6], boxes[3], boxes[7]])
    anchors = np.round(anchors)

    return torch.as_tensor(anchors)


def inclined_nms(boxes, scores, iou_threshold, max_output_size=2000):
    """
    :param  boxes format =[[x, y, w, h, theta], [x, y, w, h, theta], ...,[x, y, w, h, theta]]
    :param  scores format = [score1, score2, score3, ... , scoren]
    :param  iou_threshold format 0.2 ~ 0.4
    :return: keep index of the boxes and scores, format[0, 1, 15, 20, ... ,n]
    """
    if not isinstance(boxes, np.ndarray):
        np.array(boxes, dtype=np.float32)
    ## 旋转的nms计算
    ##输入box是5tuple
    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 1e-5)

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)
    # return np.array(keep, dtype=np.int64)


def inclined_iou(ex_boxes, gt_boxes):
    """
    :param  ex_boxes format =[[x, y, w, h, theta], [x, y, w, h, theta], ...,[x, y, w, h, theta]]
    :param  gt_boxes format =[[x, y, w, h, theta], [x, y, w, h, theta], ...,[x, y, w, h, theta]]
    :return: keep index of the boxes and scores, format[[iou00], [iou01], ..., [iou51], ... ,[iounm]]
            shape = (len(gt_boxes), len(ex_boxes))
    """
    if not isinstance(ex_boxes, np.ndarray):
        np.array(ex_boxes, dtype=np.float32)
    if not isinstance(gt_boxes, np.ndarray):
        np.array(gt_boxes, dtype=np.float32)
    ex_area = ex_boxes[:, 2] * ex_boxes[:, 3]
    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
    overlaps = []
    for i, box1 in enumerate(gt_boxes):
        temp_overlaps = []
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        for j, box2 in enumerate(ex_boxes):
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (ex_area[j] + gt_area[i] - int_area) + 1e-5

                temp_overlaps.append(inter)

            else:
                temp_overlaps.append(0.0)

        overlaps.append(temp_overlaps)

    # return np.array(overlaps, dtype=np.float32)
    return torch.as_tensor(overlaps)


if __name__ == '__main__':
    # boxes = torch.as_tensor([[25, 25, 50, 50, 0], [20, 20, 60, 60, 0]], dtype=torch.float32)
    # boxes = convert4(boxes, mode='xywh')
    # print(boxes)
    # # boxes = [[0, 0, 0, 4, 5, 4, 5, 0], [0, 0, 0, 10, 10, 10, 10, 0]]
    image = 255 * np.ones((512, 512), dtype=np.uint8)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    boxes = np.array([[100, 100, 170, 220]])
    cv2.rectangle(image, tuple(boxes[0][:2]), tuple(boxes[0][2:]), (0))

    cv2.circle(image, tuple((int(boxes[0][0] / 2 + boxes[0][2] / 2),
                             int(boxes[0][1] / 2 + boxes[0][3] / 2))
                            ), 1, (0)
               )
    boxes_r = rotated_anchors(boxes, [45, 30, -30, -45]).numpy()
    for box in boxes_r:
        cv2.polylines(image, np.array(box.reshape(1, 4, 2), dtype=np.int32), True, (230))

    boxes5pi = encode(boxes_r).numpy()
    print(boxes5pi)
    # boxes_conv4 = convert4(boxes5pi).cpu().numpy()
    # for box4 in boxes_conv4:
    #     cv2.rectangle(image, tuple(box4[:2]), tuple(box4[2:]), (1))

    # boxes_conv8 = convert8(boxes5pi).cpu().numpy()
    # for box in boxes_conv8:
    #     cv2.polylines(image, np.array(box.reshape(1, 4, 2), dtype=np.int32), True, (230))

    cv2.imshow('image', image)
    cv2.imwrite('/home/gzh/rotate_anchor.jpg', image)
    cv2.waitKey(0)

    # box = wh2xy(decode(boxes5pi))
    # boxes5a = decode(boxes5pi)
    # # boxesxy = wh2xy(boxes5a)
    # print('xyxy 4', boxes)
    # print('(xyxyxyxy)8:', boxes_r)
    # print('(xywha(pi))5', boxes5pi)
    # print('(xyxy)4 conv4', boxes_conv4)
    # # print('(xywha(cv))5', boxes5a)
    # # print('(xyxyxyxy)8 from decode:', boxesxy)
    # # print('box', box)
    # # boxes = quad2inrect(boxes_r)
    # # print('mode=xywht\n', boxes)
    #
    # # boxes = wh2xy(boxes)
    # # print('mode=xyxyxyxy\n', boxes)
    #
    # # boxes = xy2wh(boxes)
    # # print(boxes)
