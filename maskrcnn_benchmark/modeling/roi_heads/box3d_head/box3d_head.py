# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

import numpy as np
import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList
from .inference import make_roi_box3d_post_processor
from .loss import make_roi_box3d_loss_evaluator
from .roi_box3d_feature_extractors import make_roi_box3d_feature_extractor
from .roi_box3d_pc_feature_extractors import make_roi_pc_feature_extractor
from .roi_box3d_predictors import make_roi_box3d_predictor
from .roi_box3d_predictors_dim import make_roi_box3d_predictor_dimension
from .roi_box3d_predictors_loc_pc import make_roi_box3d_predictor_localization_pc
from .roi_box3d_predictors_loc_roi import make_roi_box3d_predictor_localization_conv
from .roi_box3d_predictors_rot_conf import make_roi_box3d_predictor_rotation_confidences
from .roi_box3d_predictors_rot_reg import make_roi_box3d_predictor_rotation_regression


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
        self.pc_feature_extractor = make_roi_pc_feature_extractor(cfg)
        self.predictor = make_roi_box3d_predictor(cfg)
        self.predictor_dimension = make_roi_box3d_predictor_dimension(cfg)
        self.predictor_rotation_confidences = make_roi_box3d_predictor_rotation_confidences(cfg)
        self.predictor_rotation_angle_sin_add_cos = make_roi_box3d_predictor_rotation_regression(cfg)
        self.predictor_localization_conv = make_roi_box3d_predictor_localization_conv(cfg)
        self.predictor_localization_pc = make_roi_box3d_predictor_localization_pc(cfg)
        self.post_processor = make_roi_box3d_post_processor(cfg)
        self.loss_evaluator = make_roi_box3d_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None, img_original_ids=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.
            img_original_ids (list[str]): the image original filename index
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

        pc_features = self.pc_feature_prepare(proposals, img_original_ids)
        pc_features = torch.cat(pc_features)
        fusion_feature = torch.cat((x, pc_features), 1)

        roi_fusion_feature = self.predictor(fusion_feature)

        box3d_dim_regression = self.predictor_dimension(roi_fusion_feature)
        box3d_rotation_logits = self.predictor_rotation_confidences(roi_fusion_feature)
        box3d_rotation_regression = self.predictor_rotation_angle_sin_add_cos(roi_fusion_feature)
        box3d_localization_conv_regression = self.predictor_localization_conv(roi_fusion_feature)
        box3d_localization_pc_regression = self.predictor_localization_pc(pc_features)

        if not self.training:
            post_processor_list = [box3d_dim_regression, box3d_rotation_logits, box3d_rotation_regression,
                                   box3d_localization_conv_regression, box3d_localization_pc_regression]
            post_processor_tuple = tuple(post_processor_list)
            result = self.post_processor(post_processor_tuple, proposals)
            return x, result, {}

        loss_box3d_dim, loss_box3d_rot_conf, loss_box3d_rot_reg, loss_box3d_localization = self.loss_evaluator(
            proposals,
            box3d_dim_regression=box3d_dim_regression,
            box3d_rotation_logits=box3d_rotation_logits,
            box3d_rotation_regression=box3d_rotation_regression,
            box3d_localization_conv_regression=box3d_localization_conv_regression,
            box3d_localization_pc_regression=box3d_localization_pc_regression,
            targets=targets)

        loss_dict = dict()
        loss_dict["loss_box3d_dim"] = loss_box3d_dim
        loss_dict["loss_box3d_rot_conf"] = loss_box3d_rot_conf
        loss_dict["loss_box3d_rot_reg"] = loss_box3d_rot_reg
        loss_dict["loss_box3d_loc_reg"] = loss_box3d_localization

        return x, all_proposals, loss_dict

    def pc_feature_prepare(self, proposals, img_original_ids):
        pseudo_pc_features = []
        PTH = "/home/abby/Repositories/maskrcnn-benchmark/datasets/kitti/object/training/pseudo_pc"
        for proposal_per_image, img_ori_id in zip(proposals, img_original_ids):
            pseudo_pc_path = os.path.join(PTH, img_ori_id + ".npz")
            pseudo_pc = np.load(pseudo_pc_path)
            pseudo_pc = pseudo_pc['pseudo_pc']
            assert (pseudo_pc.shape[1], pseudo_pc.shape[2] == proposal_per_image.size[1], proposal_per_image.size[0]), \
                "{}, {}".format(pseudo_pc.shape, proposal_per_image.size)
            device = proposal_per_image.bbox.device
            pseudo_pc = torch.as_tensor(pseudo_pc, device=device)
            pseudo_pc_feature = self.pc_feature_extractor(proposal_per_image, pseudo_pc)
            pseudo_pc_features.append(pseudo_pc_feature)
        return pseudo_pc_features


def build_roi_box3d_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBox3DHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBox3DHead(cfg)
