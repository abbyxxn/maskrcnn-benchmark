# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn


class FastRCNNPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred

class RotationConfidencePredictor(nn.Module):
    def __init__(self, cfg):
        super(RotationConfidencePredictor, self).__init__()
        num_bins = cfg.MODEL.ROI_BOX3D_HEAD.ROTATION_BIN
        # TODO check MODEL.BACKBONE.OUT_CHANNELS = 256, but multibin need output 7*7*512
        # input_size = (cfg.MODEL.BACKBONE.OUT_CHANNELS + cfg.MODEL.ROI_BOX3D_HEAD.POINTCLOUD_OUT_CHANNELS) * (cfg.MODEL.ROI_BOX3D_HEAD.POOLER_RESOLUTION ** 2)
        # representation_size = cfg.MODEL.ROI_BOX3D_HEAD.PREDICTORS_ROTATION_CONFIDENCES_HEAD_DIM
        # self.fc6 = nn.Linear(input_size, representation_size)
        # self.lrelu = nn.LeakyReLU(0.1)
        # self.dropout = nn.Dropout(p=0.5)
        # for l in [self.fc6, ]:
        #     nn.init.kaiming_uniform_(l.weight, a=1)
        #     nn.init.constant_(l.bias, 0)
        input_size = cfg.MODEL.ROI_BOX3D_HEAD.PREDICTORS_HEAD_DIM

        self.bbox3d_rotation_conf_score = nn.Linear(input_size, num_bins)
        nn.init.normal_(self.bbox3d_rotation_conf_score.weight, std=0.001)
        for l in [self.bbox3d_rotation_conf_score, ]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        # x = self.fc6(x)
        # x = self.lrelu(x)
        # x = self.dropout(x)
        scores = self.bbox3d_rotation_conf_score(x)

        return scores


_ROI_BOX3D_PREDICTOR = {
    "FastRCNNPredictor": FastRCNNPredictor,
    "RotationConfidencePredictor": RotationConfidencePredictor,
}


def make_roi_box3d_predictor_rotation_confidences(cfg):
    func = _ROI_BOX3D_PREDICTOR[cfg.MODEL.ROI_BOX3D_HEAD.PREDICTOR_ROTATION_CONFIDENCES]
    return func(cfg)
