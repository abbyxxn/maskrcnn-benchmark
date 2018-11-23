# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.box3d_coder import Box3dCoder
from maskrcnn_benchmark.modeling.orientation_coder import OrientationCoder

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self, box3d_coder=None, orientation_coder=None
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        # self.score_thresh = score_thresh
        # self.nms = nms
        # self.detections_per_img = detections_per_img
        if box3d_coder is None:
            box3d_coder = Box3dCoder(weights=(5., 5., 5, 10., 10., 10))
        self.box3d_coder = box3d_coder
        self.orientation_coder = orientation_coder

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        box3d_dim_regression, box3d_rotation_logits, box3d_rotation_regression, \
        box3d_localization_conv_regression, box3d_localization_pc_regression = x

        num_box3d = box3d_dim_regression.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_box3d, device=labels.device)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        # boxes_3d = [a.get_field("boxes_3d") for a in boxes]
        # concat_boxes_3d = torch.cat([a.bbox_3d for a in boxes_3d], dim=0)

        labels = labels - 1
        map_inds = 3 * labels.cpu()[:, None] + torch.tensor([0, 1, 2])
        pred_box_3d_dim = self.box3d_coder.decode(
            box3d_dim_regression[index[:, None], map_inds], labels
        )

        pred_box_3d_orientation = self.orientation_coder.decode(
            box3d_rotation_logits, box3d_rotation_regression
        )


        num_classes = box3d_dim_regression.shape[1] / 3
        # proposals = self.box_coder.decode(
        #     box_regression.view(sum(boxes_per_image), -1), concat_boxes
        # )

        # num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results
       # class_logits, box_regression = x
       # class_prob = F.softmax(class_logits, -1)

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms, score_field="scores"
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_roi_box3d_post_processor(cfg):
    # use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox3d_reg_weights = cfg.MODEL.ROI_BOX3D_HEAD.BBOX3D_REG_WEIGHTS
    box3d_coder = Box3dCoder(weights=bbox3d_reg_weights)
    # bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    # box_coder = BoxCoder(weights=bbox_reg_weights)

    # score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    # nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    # detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    orientation_coder = OrientationCoder(cfg.MODEL.ROI_BOX3D_HEAD.ROTATION_BIN,
                                         cfg.MODEL.ROI_BOX3D_HEAD.ROTATION_OVERLAP)

    postprocessor = PostProcessor(
        box3d_coder, orientation_coder
    )
    return postprocessor


if __name__ == "__main__":
    box3d_coder = Box3dCoder(weights=(5., 5., 5, 10., 10., 10))
    orientation_coder = OrientationCoder(2, 0.1)
    postprocessor = make_roi_box3d_post_processor(box3d_coder, orientation_coder)

    postprocessor(x, boxes)