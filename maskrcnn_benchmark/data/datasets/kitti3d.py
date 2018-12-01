# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Simple dataset class that wraps a list of path names
"""

import _pickle as cPickle
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.bounding_box_3d import Box3List

TYPICAL_DIMENSION = {}


class KITTIDataset(data.Dataset):
    def __init__(self, root, ann_file, remove_images_without_annotations, transforms=None):
        super(KITTIDataset, self).__init__()
        # from pycocotools.coco import COCO
        # self.coco = COCO(ann_file)
        self.image_index, self.label_list, self.boxes_list, self.boxes_3d_list, self.alphas_list = self.get_pkl_element(
            ann_file)
        self.typical_dimension = self.get_typical_dimension(self.label_list, self.boxes_3d_list)
        self.set_typical_dimension(self.typical_dimension)
        # self.boxes_3d_list = self.boxes3d_encode(self.boxes_3d_list, self.label_list)
        number_image = len(self.image_index)
        self.image_lists = []
        self.calib_lists = []
        self.depth_list = []
        for i in range(number_image):
            self.image_lists.append(root + '/training' + '/image_2/' + self.image_index[i] + "_01.png")
            self.calib_lists.append(root + '/training' + '/calib/' + self.image_index[i] + ".txt")
            self.depth_list.append(root + '/training' + '/depth/' + self.image_index[i] + "_01.png.npz")
        self.transforms = transforms
        self.id_to_img_map = self.image_index
        cache_file = os.path.join(root, 'typical_dimension_gt.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(self.typical_dimension, fid)
        print('wrote typical dimension gt to {}'.format(cache_file))
        # self.alphas_list2 = self.get_alpha(self.calib_lists, self.boxes_list)
        # alphas = []
        # for i, alpha in enumerate(self.alphas_list2):
        #     alphas.append((self.alphas_list[i] - alpha))

        # TODO implement remove_images_without_annotations:
        if remove_images_without_annotations:
            pass

    def __getitem__(self, idx):
        img = Image.open(self.image_lists[idx]).convert("RGB")
        boxes = self.boxes_list[idx]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, img.size, mode="xyxy")

        classes = self.label_list[idx]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        boxes_3d = self.boxes_3d_list[idx]
        boxes_3d = torch.as_tensor(boxes_3d).reshape(-1, 7)
        boxes_3d = Box3List(boxes_3d, img.size)
        target.add_field("boxes_3d", boxes_3d)

        alphas = self.alphas_list[idx]
        alphas = torch.tensor(alphas)
        num_instances = alphas.shape[0]
        target.add_field("alphas", alphas)

        # depth = self.image_index[idx]
        # depths = []
        # for i in range(num_instances):
        #     depths.append(int(depth))
        # depths = torch.tensor(depths)
        # target.add_field("depth", depths)


        d = np.load(self.depth_list[idx])
        # depth = np.transpose(d['depths'])
        depth = d['depths']
        # assert depth.shape == img.size, "{}, {}".format(
        #     depth.shape, img.size
        #     )
        depths = []
        for i in range(num_instances):
            depths.append(depth)
        depths = torch.tensor(depths)
        target.add_field("depth", depths)

        # TODO clip
        target = target.clip_to_image(remove_empty=True)
        # dummy target
        # w, h = img.size
        # target = BoxList([[0, 0, w, h]], img.size, mode="xyxy")

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        img_original_idx = self.id_to_img_map[idx]

        return img, target, idx, img_original_idx

    def __len__(self):
        return len(self.image_lists)

    def get_alpha(self, calib_lists, boxes_list):
        alpha_list = []
        for i, filename in enumerate(calib_lists):
            with open(filename, 'r') as f:
                calib = {}
                for line in f:
                    fields = line.split()
                    if len(fields) is 0:
                        continue
                    key = fields[0][:-1]
                    val = np.asmatrix(fields[1:]).astype(np.float32).reshape(3, -1)
                    calib[key] = val
                fx = calib['P2'][0, 0]
                cx = calib['P2'][0, 2]
                x = (boxes_list[i][:, 0] + boxes_list[i][:, 2]) / 2
                alpha_list.append(np.arctan2(x - cx, fx))
        return alpha_list




    def get_img_info(self, idx):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        img = Image.open(self.image_lists[idx]).convert("RGB")
        width, height = img.size
        return {"height": height, "width": width}

    @staticmethod
    def get_pkl_element(ann_file):
        '''
        labels mapping:
        1 : person_sitting, pedestrian  pedestrian
        2 : cyclist, riding             cyclist
        3 : van, car                    car
        4 : train, bus
        5 : truck
        :param ann_file:
        :return:
        '''
        image_original_index = {}
        labels = {}
        boxes_list = {}
        boxes_3d_list = {}
        alphas_list = {}
        index = 0
        if os.path.exists(ann_file):
            with open(ann_file, 'rb') as file:
                roidb = cPickle.load(file)
                for roi in roidb:
                    image_original_index[index] = roi['image_original_index']
                    labels[index] = roi['label']
                    boxes_list[index] = roi['boxes']
                    boxes_3d_list[index] = roi['boxes_3d']
                    alphas_list[index] = roi['alphas']
                    index = index + 1
        return image_original_index, labels, boxes_list, boxes_3d_list, alphas_list

    @staticmethod
    def get_typical_dimension(label_list, boxes_3d_list):
        typical_dimension = {}
        categories = {}
        for index, label in label_list.items():
            for i, boxes_3d in enumerate(boxes_3d_list[index]):
                value = typical_dimension.get(label[i], [0, 0, 0])
                count = categories.get(label[i], 0)
                value = value + boxes_3d[1:4]
                count = count + 1
                typical_dimension[label[i]] = value
                categories[label[i]] = count
        result = {}
        for k, v in typical_dimension.items():
            result[k] = v / categories[k]

        return result  # lhw

    @staticmethod
    def set_typical_dimension(typical_dimension):
        global TYPICAL_DIMENSION
        TYPICAL_DIMENSION = typical_dimension
        return


