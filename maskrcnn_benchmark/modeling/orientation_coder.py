# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch
import numpy as np
PI = 3.14159

class OrientationCoder(object):
    """
    This class encodes and decodes a set of 3d bounding boxes orientation into
    the representation used for training the regressors.
    """

    def __init__(self, num_bins, overlap):
        """
        Arguments:
            num_bins (int)
            overlap (float)
        """
        self.num_bins = num_bins
        self.overlap = overlap

    def encode(self, reference_orientation):
        """
        Encode a set of reference orientation

        Arguments:
            reference_orientation (Tensor): reference boxes
        """
        if len(reference_orientation) == 0:
            return torch.empty(0, dtype=torch.float32)
        orientations = torch.zeros(reference_orientation.shape[0], self.num_bins, 2)
        confidences = torch.zeros(reference_orientation.shape[0], self.num_bins)
        for i, alpha in enumerate(reference_orientation):
            orientation = torch.zeros(self.num_bins, 2)
            confidence = torch.zeros(self.num_bins)
            anchors = self.compute_anchors(alpha)
            for anchor in anchors:
                orientation[anchor[0], :] = torch.tensor([torch.cos(anchor[1]), torch.sin(anchor[1])])
                confidence[anchor[0]] = 1.
            # confidence = confidence / torch.sum(confidence)
            orientations[i, :, :] = orientation
            confidences[i, :] = confidence
        return confidences, orientations


    def angle_from_multibin(angle_conf, angle_loc, overlap):
        num_bins = angle_conf.shape[1]
        bins = np.zeros((num_bins, 2), dtype=np.float32)
        bin_angle = 2 * PI / num_bins + overlap
        start = -PI - overlap / 2
        for i in range(num_bins):
            bins[i, 0] = start
            bins[i, 1] = start + bin_angle
            start = bins[i, 1] - overlap

        alphas = np.zeros((angle_conf.shape[0],), dtype=np.float32)
        for k in range(angle_conf.shape[0]):
            bin_ctrs = ((bins[:, 0] + bins[:, 1]) / 2).reshape(1, -1)  # 1 x num_bins
            conf_ctr = bin_ctrs[0, np.argmax(angle_conf[k, :].reshape(1, -1))]
            ind = np.argmax(angle_conf[k, :])
            cos_alpha = angle_loc[k, 2 * ind]
            sin_alpha = angle_loc[k, 2 * ind + 1]
            loc_alpha = np.arctan2(sin_alpha, cos_alpha)
            alphas[k] = conf_ctr + loc_alpha

        return alphas

    def compute_anchors(self, alpha):
        alpha = alpha + PI / 2
        if alpha < 0:
            alpha = alpha + 2. * PI
        alpha = alpha - int(alpha / (2. * PI)) * (2. * PI)
        anchors = []
        wedge = 2. * PI / self.num_bins

        l_index = int(alpha / wedge)
        r_index = l_index + 1
      #  r_index = r_index % self.num_bins

        if l_index * wedge - self.overlap / 2 < alpha < r_index * wedge + self.overlap / 2:
            anchors.append([l_index, alpha - (l_index+r_index)*wedge/2 ])

        if alpha < l_index * wedge + self.overlap / 2:
            alpha = alpha - ((l_index-1)%self.num_bins * wedge + wedge/2)
            if alpha > wedge:
                alpha = 2 * PI - alpha
            anchors.append([(l_index-1)%self.num_bins, alpha])

        if alpha > r_index * wedge - self.overlap / 2:
            alpha = alpha - (r_index%self.num_bins * wedge + wedge/2)
            if alpha > wedge:
                alpha = 2*PI - alpha
            anchors.append([r_index%self.num_bins, alpha])

        return anchors

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes
