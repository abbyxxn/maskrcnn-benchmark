# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_box3d_feature_extractors import make_roi_box3d_feature_extractor
from .roi_box3d_predictors_dimension import make_roi_box3d_predictor_dimension
from .roi_box3d_predictors_rotation_confidences import make_roi_box3d_predictor_rotation_confidences
from .roi_box3d_predictors_rotation_angle_sin_add_cos import make_roi_box3d_predictor_rotation_angle_sin_add_cos
from .inference import make_roi_box3d_post_processor
from .loss import make_roi_box3d_loss_evaluator


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds

class ROIBox3DHead(torch.nn.Module):
    """
    Generic Box3d Head class.
    """

    # TODO change rotation_angle_sin_add_cos to rotation_regression
    def __init__(self, cfg):
        super(ROIBox3DHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_box3d_feature_extractor(cfg)
        self.predictor_dimension = make_roi_box3d_predictor_dimension(cfg)
        self.predictor_rotation_confidences = make_roi_box3d_predictor_rotation_confidences(cfg)
        self.predictor_rotation_angle_sin_add_cos = make_roi_box3d_predictor_rotation_angle_sin_add_cos(cfg)
        self.post_processor = make_roi_box3d_post_processor(cfg)
        self.loss_evaluator = make_roi_box3d_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)

        if self.training and self.cfg.MODEL.ROI_BOX3D_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)

        box3d_dim_regression = None
        box3d_rotation_logits = None
        box3d_rotation_regression = None

        if self.cfg.MODEL.BOX3D_DIMENSION_ON:
            box3d_dim_regression = self.predictor_dimension(x)

        if self.cfg.MODEL.BOX3D_ROTATION_CONFIDENCES_ON:
            box3d_rotation_logits = self.predictor_rotation_confidences(x)

        if self.cfg.MODEL.BOX3D_ROTATION_REGRESSION_ON:
            box3d_rotation_regression = self.predictor_rotation_angle_sin_add_cos(x)
        # TODO optimizer split train box3d head

        if not self.training:
            post_processor_list = []
            if self.cfg.MODEL.BOX3D_DIMENSION_ON:
                post_processor_list.append(box3d_dim_regression)
            if self.cfg.MODEL.BOX3D_ROTATION_CONFIDENCES_ON:
                post_processor_list.append(box3d_rotation_logits)
            if self.cfg.MODEL.BOX3D_ROTATION_REGRESSION_ON:
                post_processor_list.append(box3d_rotation_regression)
            post_processor_tuple = tuple(post_processor_list)
            result = self.post_processor(post_processor_tuple, proposals)
            return x, result, {}

        loss_box3d_dim, loss_box3d_rot_conf, loss_box3d_rot_reg = self.loss_evaluator(proposals,
                                                    box3d_dim_regression=box3d_dim_regression,
                                                    box3d_rotation_logits=box3d_rotation_logits,
                                                    box3d_rotation_regression=box3d_rotation_regression,
                                                    targets=targets)

        loss_dict = {}
        if self.cfg.MODEL.BOX3D_DIMENSION_ON:
            loss_dict["loss_box3d_dim"] = loss_box3d_dim
        if self.cfg.MODEL.BOX3D_ROTATION_CONFIDENCES_ON:
            loss_dict["loss_box3d_rot_conf"] = loss_box3d_rot_conf
        if self.cfg.MODEL.BOX3D_ROTATION_REGRESSION_ON:
            loss_dict["loss_box3d_rot_reg"] = loss_box3d_rot_reg

        return (
            x,
            all_proposals,
            loss_dict,
        )


def build_roi_box3d_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBox3DHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBox3DHead(cfg)
