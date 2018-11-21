# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.orientation_coder import OrientationCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
import numpy as np


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, proposal_matcher):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels", "depth"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        # boxes3d_targets = []
        # confidences = []
        # orientations = []
        disparities = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            # bounding_box_3d = matched_targets.get_field("boxes_3d")
            # bounding_box_3d = bounding_box_3d[positive_inds]

            disparity = matched_targets.get_field("depth")
            disparity = disparity[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            # assert bounding_box_3d.size == positive_proposals.size, "{}, {}".format(
            #     bounding_box_3d, positive_proposals
            # )

            # confidences_per_image, orientations_per_image = self.orientation_coder.encode(alpha_orientation)
            # confidences_per_image, orientations_per_image, boxes3d_per_image = self.boxes3d_encode(
            #     bounding_box_3d.bbox_3d, positive_proposals.get_field("labels")
            # )
            #
            # labels.append(labels_per_image)
            # boxes3d_targets.append(boxes3d_per_image)
            # confidences.append(confidences_per_image)
            # orientations.append(orientations_per_image)
            # bounding_box_3d_box = torch.as_tensor(bounding_box_3d.bbox_3d, dtype=torch.float32)
            # confidences_per_image = torch.tensor(confidences_per_image, dtype=torch.float32)
            # orientations_per_image = torch.tensor(orientations_per_image, dtype=torch.float32)
            # confidences.append(confidences_per_image)
            # orientations.append(orientations_per_image)
            labels.append(labels_per_image)
            disparities.append(disparity)
            # boxes3d_targets.append(bounding_box_3d.bbox_3d)

        return labels, disparities

        # return labels, boxes3d_targets, confidences, orientations

    def __call__(self, proposals, box3d_dim_regression,
                 box3d_rotation_logits,
                 box3d_rotation_regression,
                 targets):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        # labels, boxes3d_targets, confidences_targets, orientations_targets = self.prepare_targets(proposals, targets)
        labels, boxes3d_targets, confidences_targets, orientations_targets = self.prepare_targets(proposals, targets)
        labels = cat(labels, dim=0)
        boxes3d_targets = cat(boxes3d_targets, dim=0)
        confidences_targets = cat(confidences_targets, dim=0)
        orientations_targets = cat(orientations_targets, dim=0)
        # positive_inds = torch.nonzero(labels > 0).squeeze(1)
        # labels_pos = labels[positive_inds]
        labels = labels - 1
        map_inds = 3 * labels.cpu()[:, None] + torch.tensor([0, 1, 2])

        box3d_loss = None
        rotation_confidence_loss = None
        rotation_regression_loss = None


        # box3d_dim_regression = cat(box3d_dim_regression, dim=0)
        # device = box3d_dim_regression.device if isinstance(box3d_dim_regression, torch.Tensor) else torch.device("cpu")
        # boxes3d_targets = torch.as_tensor(boxes3d_targets, dtype=torch.float32, device=device)

        # box3d_dim_regression = cat(box3d_dim_regression, dim=0)
        device = box3d_dim_regression.device
        boxes3d_targets = torch.as_tensor(boxes3d_targets, dtype=torch.float32, device=device)
        box3d_loss = smooth_l1_loss(
            box3d_dim_regression[:, map_inds],
            boxes3d_targets[:, 1:4],
            size_average=False,
            beta=1,
        )
        box3d_loss = box3d_loss / labels.numel()
        box3d_loss = box3d_loss / 10000


        # box3d_rotation_logits = torch.as_tensor(box3d_rotation_logits, dtype=torch.float32, device=device)
        # box3d_rotation_logits = cat(box3d_rotation_logits, dim=0)
        confidences_targets = torch.as_tensor(confidences_targets, dtype=torch.float32, device=device)
        # rotation_confidence_loss = torch.nn.CrossEntropyLoss(box3d_rotation_logits, confidences_targets)
        rotation_confidence_loss = self.confidence_loss(box3d_rotation_logits, confidences_targets)

        # rotation_confidence_loss = torch.tensor(rotation_confidence_loss, dtype=torch.float32, device=device)

        # box3d_rotation_regression = torch.as_tensor(box3d_rotation_regression, dtype=torch.float32, device=device)

        # box3d_rotation_regression = cat(box3d_rotation_regression, dim=0)
        orientations_targets = torch.as_tensor(orientations_targets, dtype=torch.float32, device=device)
        box3d_rotation_regression = box3d_rotation_regression.reshape(-1, self.num_bins, 2)
        box3d_rotation_regression = F.normalize(box3d_rotation_regression, dim=2)
        rotation_regression_loss = self.orientation_loss(orientations_targets, box3d_rotation_regression)
        rotation_regression_loss = rotation_regression_loss / 10
        # rotation_regression_loss = torch.sum(box3d_rotation_regression) / 100
        # rotation_regression_loss = torch.tensor(rotation_regression_loss, dtype=torch.float32, device=device)
        # if boxes3d_targets.numel() == 0:
        #     return boxes3d_targets.sum() * 0
        # box3d_loss = torch.as_tensor(box3d_loss, dtype=torch.float32, device=device)
        # rotation_confidence_loss = torch.as_tensor(rotation_confidence_loss, dtype=torch.float32, device=device)
        # rotation_regression_loss = torch.as_tensor(rotation_regression_loss, dtype=torch.float32, device=device)
        # TODO check classification need be divide by labels_pos.numel()
        return box3d_loss, rotation_confidence_loss, rotation_regression_loss





def make_roi_box3d_localization_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    # orientation_coder = OrientationCoder(cfg.MODEL.ROI_BOX3D_HEAD.ROTATION_BIN, cfg.MODEL.ROI_BOX3D_HEAD.ROTATION_OVERLAP)
    loss_evaluator = FastRCNNLossComputation(matcher)
    return loss_evaluator
