# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import os

import torch
import _pickle as cPickle


class Box3dCoder(object):
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

    def encode(self, reference_boxes, labels):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        device = reference_boxes.device

        cache_file = os.path.join('/home/jiamingsun/raid/dataset/kitti/object', 'typical_dimension_gt.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as file:
                typical_dimension = cPickle.load(file)

        ex_lengths = torch.zeros(reference_boxes.shape[0], dtype=torch.float32, device=device)
        ex_heights = torch.zeros(reference_boxes.shape[0], dtype=torch.float32, device=device)
        ex_widths = torch.zeros(reference_boxes.shape[0], dtype=torch.float32, device=device)

        for i, label in enumerate(labels):
            ex_lengths[i], ex_heights[i], ex_widths[i] = typical_dimension[label.item()]

        gt_ry = reference_boxes[:, 0]
        gt_lengths = reference_boxes[:, 1]
        gt_heights = reference_boxes[:, 2]
        gt_widths = reference_boxes[:, 3]
        gt_ctr_x = reference_boxes[:, 4]
        gt_ctr_y = reference_boxes[:, 5]
        gt_ctr_z = reference_boxes[:, 6]

        wl, wh, ww, wx, wy, wz = self.weights

        targets_ry = gt_ry
        targets_dx = gt_ctr_x
        targets_dy = gt_ctr_y
        targets_dz = gt_ctr_z

        targets_dl = wl * torch.log(gt_lengths / ex_lengths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets_dw = ww * torch.log(gt_widths / ex_widths)

        targets = torch.stack((targets_ry, targets_dl, targets_dh, targets_dw, targets_dx, targets_dy, targets_dz),
                              dim=1)
        return targets
        # TO_REMOVE = 1  # TODO remove
        # ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        # ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        # ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        # ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights
        # gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        # gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights
        # targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        # targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights

    def decode(self, reference_boxes_3d, labels):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes prediction
            boxes (Tensor): reference boxes. proposal
        """
        device = reference_boxes_3d.device
        cache_file = os.path.join('/home/jiamingsun/raid/dataset/kitti/object', 'typical_dimension_gt.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as file:
                typical_dimension = cPickle.load(file)

        ex_lengths = torch.zeros(reference_boxes_3d.shape[0], dtype=torch.float32, device=device)
        ex_heights = torch.zeros(reference_boxes_3d.shape[0], dtype=torch.float32, device=device)
        ex_widths = torch.zeros(reference_boxes_3d.shape[0], dtype=torch.float32, device=device)

        for i, label in enumerate(labels):
            ex_lengths[i], ex_heights[i], ex_widths[i] = typical_dimension[label.item()]

        # boxes = boxes.to(rel_codes.dtype)

        # TO_REMOVE = 1  # TODO remove
        # widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        # heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        # ctr_x = boxes[:, 0] + 0.5 * widths
        # ctr_y = boxes[:, 1] + 0.5 * heights

        wl, wh, ww, wx, wy, wz = self.weights
        dl = reference_boxes_3d[:, 1] / wl
        dh = reference_boxes_3d[:, 2] / wh
        dw = reference_boxes_3d[:, 3] / ww

        # Prevent sending too large values into torch.exp()
        dl = torch.clamp(dl, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        dw = torch.clamp(dw, max=self.bbox_xform_clip)

        pred_l = torch.exp(dl) * ex_lengths
        pred_h = torch.exp(dh) * ex_heights
        pred_w = torch.exp(dw) * ex_widths

        pred_boxes = torch.stack((pred_l, pred_h, pred_w),dim=1)

        # pred_boxes = torch.zeros_like(rel_codes)
        # # x1
        # pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # # y1
        # pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        # pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        # pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes