# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from maskrcnn_benchmark.utils import rotate_utils as trans


# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4 and bbox.size(-1) != 8:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4 or 8, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh", "xy8", "xy854"):
            raise ValueError("mode should be 'xyxy' or 'xywh' or 'xy8' or 'xy854'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh", "xy8", "xy854"):
            raise ValueError("mode should be 'xyxy' or 'xywh' or 'xy8' or 'xy854'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        if mode == "xyxy" or mode == "xywh":
            xmin, ymin, xmax, ymax = self._split_into_xyxy()
            if mode == "xyxy":
                bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
                bbox = BoxList(bbox, self.size, mode=mode)
            elif mode == "xywh":
                TO_REMOVE = 1
                bbox = torch.cat(
                    (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
                )
                bbox = BoxList(bbox, self.size, mode=mode)
        elif mode == "xy854":
            if isinstance(self.bbox, torch.Tensor):
                self.bbox = self.bbox
            else:
                self.bbox = torch.as_tensor(trans.quad2inrect(self.bbox))
                # self.bbox = torch.as_tensor(self.bbox)
            x1, y1, x2, y2, x3, y3, x4, y4 = self._split_into_xyxy()
            bbox8 = torch.cat(
                (x1, y1, x2, y2, x3, y3, x4, y4), dim=-1
            )
            bbox = BoxList(bbox8, self.size, mode=mode)

            gt_x = torch.cat((x1, x2, x3, x4), dim=1)
            gt_y = torch.cat((y1, y2, y3, y4), dim=1)

            gt_xmin, _ = gt_x.min(1)
            gt_xmax, _ = gt_x.max(1)
            gt_ymin, _ = gt_y.min(1)
            gt_ymax, _ = gt_y.max(1)

            bbox4 = torch.cat((gt_xmin[:, None], gt_ymin[:, None], gt_xmax[:, None], gt_ymax[:, None]), dim=1)
            bbox5 = trans.encode(bbox.bbox)
            x, y, w, h, t = bbox5.split(1, dim=-1)
            bbox5_1 = torch.cat((x, y, h, w, t - 1.5708), dim=1)

            xt, yt, xc, yc, r = (x1 + x2) / 2, (y1 + y2) / 2, x, y, torch.log(w / h)
            # xt, yt, xc, yc = (x1 + x2) / 2, (y1 + y2) / 2, x, y
            # h = torch.sqrt((xt - xc) ** 2 + (yt - yc) ** 2)
            # w = torch.sqrt((xt - x1) ** 2 + (yt - y1) ** 2)
            # r = torch.log(w / h)
            bbox_s = torch.cat((xt, yt, xc, yc, r), dim=1)

            bbox.add_field("xywht", bbox5)
            bbox.add_field("xyxy", bbox4)
            bbox.add_field("xywht1", bbox5_1)
            bbox.add_field("xyxyr", bbox_s)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        elif self.mode == "xy8":
            x1, y1, x2, y2, x3, y3, x4, y4 = self.bbox.split(1, dim=-1)
            return (
                x1,
                y1,
                x2,
                y2,
                x3,
                y3,
                x4,
                y4
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if self.mode == "xyxy" or self.mode == "xywh":
            if ratios[0] == ratios[1]:
                ratio = ratios[0]
                scaled_box = self.bbox * ratio
                bbox = BoxList(scaled_box, size, mode=self.mode)
                # bbox._copy_extra_fields(self)
                for k, v in self.extra_fields.items():
                    if not isinstance(v, torch.Tensor):
                        v = v.resize(size, *args, **kwargs)
                    bbox.add_field(k, v)
                return bbox

            ratio_width, ratio_height = ratios
            xmin, ymin, xmax, ymax = self._split_into_xyxy()
            scaled_xmin = xmin * ratio_width
            scaled_xmax = xmax * ratio_width
            scaled_ymin = ymin * ratio_height
            scaled_ymax = ymax * ratio_height
            scaled_box = torch.cat(
                (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
            )
            bbox = BoxList(scaled_box, size, mode="xyxy")
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox.convert(self.mode)

        elif self.mode == "xy8":
            ratio_width, ratio_height = ratios
            x1, y1, x2, y2, x3, y3, x4, y4 = self.bbox.split(1, dim=-1)
            scaled_x1 = x1 * ratio_width
            scaled_y1 = y1 * ratio_height
            scaled_x2 = x2 * ratio_width
            scaled_y2 = y2 * ratio_height
            scaled_x3 = x3 * ratio_width
            scaled_y3 = y3 * ratio_height
            scaled_x4 = x4 * ratio_width
            scaled_y4 = y4 * ratio_height
            scaled_bbox = torch.cat(
                (scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                 scaled_x3, scaled_y3, scaled_x4, scaled_y4
                 ), dim=-1
            )

            gt_x = torch.cat((scaled_x1, scaled_x2, scaled_x3, scaled_x4), dim=1)
            gt_y = torch.cat((scaled_y1, scaled_y2, scaled_y3, scaled_y4), dim=1)

            gt_xmin, _ = gt_x.min(1)
            gt_xmax, _ = gt_x.max(1)
            gt_ymin, _ = gt_y.min(1)
            gt_ymax, _ = gt_y.max(1)

            bbox = BoxList(scaled_bbox, size, mode="xy8")
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)

            return bbox.convert(self.mode)

        elif self.mode == "xy854":
            ratio_width, ratio_height = ratios
            x1, y1, x2, y2, x3, y3, x4, y4 = self.bbox.split(1, dim=-1)
            scaled_x1 = x1 * ratio_width
            scaled_y1 = y1 * ratio_height
            scaled_x2 = x2 * ratio_width
            scaled_y2 = y2 * ratio_height
            scaled_x3 = x3 * ratio_width
            scaled_y3 = y3 * ratio_height
            scaled_x4 = x4 * ratio_width
            scaled_y4 = y4 * ratio_height
            scaled_bbox = torch.cat(
                (scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                 scaled_x3, scaled_y3, scaled_x4, scaled_y4
                 ), dim=-1
            )

            gt_x = torch.cat((scaled_x1, scaled_x2, scaled_x3, scaled_x4), dim=1)
            gt_y = torch.cat((scaled_y1, scaled_y2, scaled_y3, scaled_y4), dim=1)

            gt_xmin, _ = gt_x.min(1)
            gt_xmax, _ = gt_x.max(1)
            gt_ymin, _ = gt_y.min(1)
            gt_ymax, _ = gt_y.max(1)

            scaled_bbox4 = torch.cat((gt_xmin[:, None], gt_ymin[:, None], gt_xmax[:, None], gt_ymax[:, None]), dim=1)
            scaled_bbox5 = trans.encode(scaled_bbox)
            x, y, w, h, t = scaled_bbox5.split(1, dim=-1)
            scaled_bbox5_1 = torch.cat((x, y, h, w, t - 1.5708), dim=1)

            xt, yt, xc, yc, r = (scaled_x1 + scaled_x2) / 2, (scaled_y1 + scaled_y2) / 2, x, y, torch.log(w / h)
            # xt, yt, xc, yc = (scaled_x1 + scaled_x2) / 2, (scaled_y1 + scaled_y2) / 2, x, y
            # h = torch.sqrt((xt - xc) ** 2 + (yt - yc) ** 2)
            # w = torch.sqrt((xt - scaled_x1) ** 2 + (yt - scaled_y1) ** 2)
            # r = torch.log(w / h)

            scaled_bbox_s = torch.cat((xt, yt, xc, yc, r), dim=1)

            bbox = BoxList(scaled_bbox, size, mode="xy854")
            self.add_field("xyxy", scaled_bbox4)
            self.add_field("xywht", scaled_bbox5)
            self.add_field("xywht1", scaled_bbox5_1)
            self.add_field("xyxyr", scaled_bbox_s)

            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)

            return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        TO_REMOVE = 1
        image_width, image_height = self.size

        if self.mode == "xyxy" or self.mode == "xywh":
            xmin, ymin, xmax, ymax = self._split_into_xyxy()
            if method == FLIP_LEFT_RIGHT:
                transposed_xmin = image_width - xmax - TO_REMOVE
                transposed_xmax = image_width - xmin - TO_REMOVE
                transposed_ymin = ymin
                transposed_ymax = ymax
            elif method == FLIP_TOP_BOTTOM:
                transposed_xmin = xmin
                transposed_xmax = xmax
                transposed_ymin = image_height - ymax
                transposed_ymax = image_height - ymin

            transposed_boxes = torch.cat(
                (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
            )
            bbox = BoxList(transposed_boxes, self.size, mode="xyxy")

            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.transpose(method)
                bbox.add_field(k, v)
            return bbox.convert(self.mode)
        elif self.mode == "xy854":
            x1, y1, x2, y2, x3, y3, x4, y4 = self.bbox.split(1, dim=-1)
            if method == FLIP_LEFT_RIGHT:
                transposed_x1 = image_width - x1 - TO_REMOVE
                transposed_x2 = image_width - x2 - TO_REMOVE
                transposed_x3 = image_width - x3 - TO_REMOVE
                transposed_x4 = image_width - x4 - TO_REMOVE
                transposed_y1 = y1
                transposed_y2 = y2
                transposed_y3 = y3
                transposed_y4 = y4

                transposed_boxes = torch.cat(
                    (transposed_x1, transposed_y1, transposed_x2, transposed_y2,
                     transposed_x3, transposed_y3, transposed_x4, transposed_y4), dim=-1
                )
                bbox = BoxList(transposed_boxes, self.size, mode=self.mode)

                gt_x = torch.cat((transposed_x1, transposed_x2, transposed_x3, transposed_x4), dim=1)
                gt_y = torch.cat((transposed_y1, transposed_y2, transposed_y3, transposed_y4), dim=1)

                gt_xmin, _ = gt_x.min(1)
                gt_xmax, _ = gt_x.max(1)
                gt_ymin, _ = gt_y.min(1)
                gt_ymax, _ = gt_y.max(1)

                transposed_bbox4 = torch.cat((gt_xmin[:, None], gt_ymin[:, None], gt_xmax[:, None], gt_ymax[:, None]),
                                             dim=1)
                transposed_bbox5 = trans.encode(transposed_boxes)
                x, y, w, h, t = transposed_bbox5.split(1, dim=-1)
                transposed_bbox5_1 = torch.cat((x, y, h, w, t - 1.5708), dim=1)

                xt, yt, xc, yc, r = (transposed_x1 + transposed_x2) / 2, (transposed_y1 + transposed_y2) / 2, x, y, torch.log(w / h)
                # xt, yt, xc, yc = (transposed_x1 + transposed_x2) / 2, (transposed_y1 + transposed_y2) / 2, x, y
                # h = torch.sqrt((xt - xc) ** 2 + (yt - yc) ** 2)
                # w = torch.sqrt((xt - transposed_x1) ** 2 + (yt - transposed_y1) ** 2)
                # r = torch.log(w / h)

                transposed_bbox_s = torch.cat((xt, yt, xc, yc, r), dim=1)

                self.add_field("xyxy", transposed_bbox4)
                self.add_field("xywht", transposed_bbox5)
                self.add_field("xywht1", transposed_bbox5_1)
                self.add_field("xyxyr", transposed_bbox_s)

                # bbox._copy_extra_fields(self)
                for k, v in self.extra_fields.items():
                    if not isinstance(v, torch.Tensor):
                        v = v.transpose(method)
                    bbox.add_field(k, v)
                return bbox.convert(self.mode)

            elif method == FLIP_TOP_BOTTOM:
                transposed_x1 = x1
                transposed_x2 = x2
                transposed_x3 = x3
                transposed_x4 = x4
                transposed_y1 = image_height - y1 - TO_REMOVE
                transposed_y2 = image_height - y2 - TO_REMOVE
                transposed_y3 = image_height - y3 - TO_REMOVE
                transposed_y4 = image_height - y4 - TO_REMOVE
                print("None")

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        if self.mode == "xyxy":
            self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
            self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
            self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
            self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
            if remove_empty:
                box = self.bbox
                keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
                return self[keep]
        elif self.mode == "xy854":
            self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
            self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
            self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
            self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
            self.bbox[:, 4].clamp_(min=0, max=self.size[0] - TO_REMOVE)
            self.bbox[:, 5].clamp_(min=0, max=self.size[1] - TO_REMOVE)
            self.bbox[:, 6].clamp_(min=0, max=self.size[0] - TO_REMOVE)
            self.bbox[:, 7].clamp_(min=0, max=self.size[1] - TO_REMOVE)
            if remove_empty:
                #                 pass
                #                 remove the zero area box
                keep = (self.get_field("xywht")[:, 2] > 1) & (self.get_field("xywht")[:, 3] > 1)
                return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        elif self.mode == "xy854":
            box = self.extra_fields["xywht"]
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)
