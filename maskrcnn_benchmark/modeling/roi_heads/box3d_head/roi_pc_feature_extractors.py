# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.modeling.poolers import Pooler

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

class Box3dPCFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(Box3dPCFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX3D_HEAD.POOLER_RESOLUTION
        scales = (1.,)
        sampling_ratio = cfg.MODEL.ROI_BOX3D_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

    def forward(self, proposal_per_image, depth):
        device = proposal_per_image.bbox.device
        # depth = target_per_image.extra_fields["depth"]
        f = width_to_focal[depth.shape[1]]
        pointcloud = self.depth_to_pointcloud(depth, f, f, depth.shape[1] / 2, depth.shape[0] / 2)
        w, h = depth.shape
        # pointcloud = pointcloud.reshape(3, w, h)
        pointcloud = torch.unsqueeze(pointcloud.reshape(3, w, h), 0)
        pointclouds = ()
        pointclouds = pointclouds + (pointcloud,)
        proposal = []
        proposal.append(proposal_per_image)
        # rand_num = torch.rand(1, 3, 1224, 370)
        # rand_num = torch.as_tensor(rand_num, dtype=torch.float32, device=device)
        # depths = depths + (rand_num,)
        x = self.pooler(pointclouds, proposal)
        return x

    # def forward(self, proposal_per_image, target_per_image):
    #     device = proposal_per_image.bbox.device
    #     depth = target_per_image.extra_fields["depth"]
    #     f = width_to_focal[depth.shape[1]]
    #     pointcloud = self.depth_to_pointcloud(depth[0], f, f, depth.shape[1]/2, depth.shape[2]/2)
    #     w, h = depth[0].shape
    #     # pointcloud = pointcloud.reshape(3, w, h)
    #     pointcloud = torch.unsqueeze(pointcloud.reshape(3, w, h), 0)
    #     pointclouds = []
    #     pointclouds = [pointcloud]
    #     proposal = [proposal_per_image]
    #     # proposal.append(proposal_per_image)
    #     # rand_num = torch.rand(1, 3, 1224, 370)
    #     # rand_num = torch.as_tensor(rand_num, dtype=torch.float32, device=device)
    #     # depths = depths + (rand_num,)
    #     x = self.pooler(pointclouds, proposal)
    #     if x.max() == 0 and x.min() == 0:
    #         print('max', pointcloud.max())
    #         print('min', pointcloud.min())
    #         print('max', depth.max())
    #         print('min', depth.min())
    #         raise AssertionError
    #     return x

    def depth_to_pointcloud(self, depth, fx, fy, cx, cy):
        device = depth.device
        fx = torch.as_tensor(fx, dtype=torch.float32, device=device)
        fy = torch.as_tensor(fy, dtype=torch.float32, device=device)
        cx = torch.as_tensor(cx, dtype=torch.float32, device=device)
        cy = torch.as_tensor(cy, dtype=torch.float32, device=device)
        w, h = depth.shape
        # w = torch.as_tensor(w, dtype=torch.float32, device=device)
        # h = torch.as_tensor(h, dtype=torch.float32, device=device)
        depth = depth.reshape(-1)
        # xx = np.linspace(0, w, w)
        # yy= np.linspace(0, h, h)
        xx = torch.arange(w)
        yy = torch.arange(h)
        x, y = torch.meshgrid(xx, yy)
        x = x.reshape(-1)
        y = y.reshape(-1)
        pointcloud = torch.zeros((3, w * h))
        pointcloud = torch.as_tensor(pointcloud, dtype=torch.float32, device=device)
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device)
        pointcloud[0, :] = depth / fx * (x - cx)
        pointcloud[1, :] = depth / fy * (y - cy)
        pointcloud[2, :] = depth
        return pointcloud


_ROI_BOX3D_FEATURE_EXTRACTORS = {
    "Box3dPCFeatureExtractor": Box3dPCFeatureExtractor,
}


def make_roi_pc_feature_extractor(cfg):
    func = _ROI_BOX3D_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX3D_HEAD.POINT_CLOUD_FEATURE_EXTRACTOR]
    return func(cfg)