# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import socket
import tempfile
import time
from collections import OrderedDict
from datetime import datetime as dt

import torch
from tqdm import tqdm

from maskrcnn_benchmark.kitti_vis import vis_2d_boxes_list, read_img
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from ..structures.bounding_box import BoxList
from ..utils.comm import is_main_process
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize


def get_run_name():
    """ A unique name for each run """
    return dt.now().strftime(
        '%b%d-%H-%M-%S') + '_' + socket.gethostname()


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in tqdm(enumerate(data_loader)):
        images, targets, image_ids = batch
        # im0 = images.tensors[0, ...]
        # im0 = transforms.functional.to_pil_image(im0)
        # im0 = numpy.asarray(im0)
        # boxes = targets[0].convert('xyxy')
        # gt_boxes = boxes.bbox.tolist()
        # img2 = vis_2d_boxes_list(im0, gt_boxes)
        # plt_show(img2)

        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
        # TODO remove
        # if i == 100:
        #     break
    return results_dict


def prepare_for_bbox3d_detection(predictions, dataset, output_folder):
    # assert isinstance(dataset, COCODataset)
    # if output_folder:
    #     output_folder = os.path.join(output_folder, "bbox3d_result")
    #     mkdir(output_folder)
    # bbox3d_results = []
    count = 0
    for image_id, prediction in enumerate(predictions):
        # TODO image_id is what
        count = count + 1
        print(count)
        idx = dataset.id_to_img_map[image_id]
        original_id = dataset.image_name[idx]
        if len(prediction) == 0:
            continue
        # bbox3d_result_path = os.path.join(output_folder, original_id + ".txt")

        # with open(bbox3d_result_path, 'w') as box3d:
        # TODO replace with get_img_info?
        # image_width = dataset.coco.imgs[original_id]["width"]
        # image_height = dataset.coco.imgs[original_id]["height"]
        image_size = dataset.get_img_info(image_id)
        # image_width = dataset.get_img_info(image_id)["width"]
        # image_height = dataset.get_img_info(image_id)["height"]
        image_width = image_size["width"]
        image_height = image_size["height"]
        # TODO resize
        # prediction = prediction.resize((image_width, image_height))
        # prediction = prediction.convert("xywh")
        prediction = prediction.convert("xyxy")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()
        boxes_3d = prediction.get_field("boxes_3d").bbox_3d.tolist()
        alphas = prediction.get_field("alphas").tolist()

        # mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        # mapped_labels = [dataset.category_id_to_label_name[i] for i in labels]

        save_kitti_3d_result(boxes, original_id, scores, output_folder, dataset, alphas, boxes_3d)
        # prediction = prediction.convert("xywh")
        # boxes = prediction.bbox.tolist()

        # for k, box in enumerate(boxes):
        #     line = [mapped_labels[k], -1, -1, alphas[k], box[0], box[1], box[2], box[3], boxes_3d[k][1],
        #             boxes_3d[k][2], boxes_3d[k][3], boxes_3d[k][4], boxes_3d[k][5], boxes_3d[k][6], boxes_3d[k][0],
        #             scores[k]]
        #     line = ' '.join([str(item) for item in line]) + '\n'
        #     box3d.write(line)

    # label_path = dataset.root + "/training/label_2"
    # result_path = output_folder
    # label_split_file_path = dataset.root + "/ImageSets/val.txt"
    # evaluate(label_path, result_path, label_split_file_path, current_class=0, coco=False, score_thresh=-1)

    return output_folder


def save_kitti_3d_result(box, original_id, scores, output_folder, dataset, alphas, boxes_3d):
    TEST_ANGLE_REG = 0
    TEST_SIZE_REG = 0
    TEST_XYZ_REG = 0
    if output_folder:
        output_folder = os.path.join(output_folder, 'detections', 'data')
        if not os.path.exists(output_folder):
            mkdir(output_folder)
    # image_index = dataset.coco.imgs[original_id]["file_name"].split('.')[0]
    filename = os.path.join(output_folder, original_id + ".txt")
    with open(filename, 'wt') as f:
        if len(box) == 0:
            return
        # boxes_3d = box
        # boxes_3d_v2 = box
        # if not TEST_ANGLE_REG:
        #     boxes_3d[:, 0] = -10
        # if not TEST_SIZE_REG:
        #     boxes_3d[:, 1:] = -1
        # if not TEST_XYZ_REG:
        #     boxes_3d_v2[:, 1:] = 1000

        for k in range(len(box)):
            height = box[k][3] - box[k][1] + 1
            if height < 25:
                continue
            ry, h, w, l, tx, ty, tz = boxes_3d[k]
            # ry, l, h, w = -10, -1, -1, -1
            # alpha = -10
            alpha = alphas[k]
            # tx, ty, tz = -1000, -1000, -1000
            f.write(
                '{:s} -1 -1 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n' \
                    .format("Car", \
                            alpha, box[k][0], box[k][1], box[k][2], box[k][3], \
                            h, w, l, tx, ty, tz, ry, scores[k]))


def prepare_for_coco_detection(predictions, dataset, output_folder):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        # TODO image_id is what
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        # image_width = dataset.coco.imgs[original_id]["width"]
        # image_height = dataset.coco.imgs[original_id]["height"]
        image_width = dataset.get_img_info(image_id)["width"]
        image_height = dataset.get_img_info(image_id)["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        mapped_labels = labels

        prediction = prediction.convert("xyxy")
        boxes = prediction.bbox.tolist()

        gt_boxes = []
        gt_ann = dataset.coco.imgToAnns[original_id]
        for item in gt_ann:
            gt_boxes.append(item['bbox'])
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), 'xywh')
        gt_boxes = gt_boxes.convert('xyxy')
        gt_boxes = gt_boxes.bbox.tolist()

        image_path = os.path.join(dataset.root, dataset.coco.imgs[original_id]["file_name"])
        img = read_img(image_path)
        save_path = os.path.join(output_folder, dataset.coco.imgs[original_id]["file_name"])
        img2 = vis_2d_boxes_list(img, boxes, gt_boxes, save_path)
        # plt_show(img2, save_path)

        save_kitti_result(boxes, original_id, scores, output_folder, dataset)
        prediction = prediction.convert("xywh")
        boxes = prediction.bbox.tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def save_kitti_result(box, original_id, scores, output_folder, dataset):
    TEST_ANGLE_REG = 0
    TEST_SIZE_REG = 0
    TEST_XYZ_REG = 0
    if output_folder:
        output_folder = os.path.join(output_folder, 'detections', 'data')
        if not os.path.exists(output_folder):
            mkdir(output_folder)
    image_index = dataset.coco.imgs[original_id]["file_name"].split('.')[0]
    filename = os.path.join(output_folder, image_index + ".txt")
    with open(filename, 'wt') as f:
        if len(box) == 0:
            return
        # boxes_3d = box
        # boxes_3d_v2 = box
        # if not TEST_ANGLE_REG:
        #     boxes_3d[:, 0] = -10
        # if not TEST_SIZE_REG:
        #     boxes_3d[:, 1:] = -1
        # if not TEST_XYZ_REG:
        #     boxes_3d_v2[:, 1:] = 1000

        for k in range(len(box)):
            height = box[k][3] - box[k][1] + 1
            if height < 25:
                continue
            ry, l, h, w = -10, -1, -1, -1
            alpha = -10
            tx, ty, tz = -1000, -1000, -1000
            f.write(
                '{:s} -1 -1 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n' \
                    .format("Car", \
                            alpha, box[k][0], box[k][1], box[k][2], box[k][3], \
                            h, w, l, tx, ty, tz, ry, scores[k]))


def prepare_for_coco_segmentation(predictions, dataset):
    import pycocotools.mask as mask_util
    import numpy as np

    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # t = time.time()
        masks = masker(masks, prediction)
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # rles = prediction.get_field('mask')

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results


# inspired from Detectron
def evaluate_box_proposals(
        predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def evaluate_predictions_on_coco(
        coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    import json

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(str(json_result_file))
    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoint": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)


def inference(
        cfg,
        model,
        data_loader,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.deprecated.get_world_size()
        if torch.distributed.deprecated.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} images".format(len(dataset)))
    start_time = time.time()
    predictions = compute_on_dataset(model, data_loader, device)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    # config MODEL WEIGHT path expect like that
    # "/raid/kitti3doutput/e2e_mask_rcnn_R_50_FPN_1x/Dec20-12-13-59_DGX-1-A7_step/model_0002500.pth"
    cfg_name = cfg.MODEL.WEIGHT.split('/')[-3]
    model_step = cfg.MODEL.WEIGHT.split('/')[-1].split('.')[0]
    output_folder = os.path.join(output_folder, cfg_name, model_step)
    if not os.path.exists(output_folder):
        mkdir(output_folder)

    # if output_folder:
    #     torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    if box_only:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return
    logger.info("Preparing results for COCO format")
    coco_results = {}
    # if "bbox" in iou_types:
    #     logger.info("Preparing bbox results")
    #     coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset, output_folder)
    if "segm" in iou_types:
        logger.info("Preparing segm results")
        coco_results["segm"] = prepare_for_coco_segmentation(predictions, dataset)
    if "bbox3d" in iou_types:
        logger.info("Preparing bbox3d results")
        coco_results["bbox3d"] = prepare_for_bbox3d_detection(predictions, dataset, output_folder)
        return

    results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            res = evaluate_predictions_on_coco(
                dataset.coco, coco_results[iou_type], file_path, iou_type
            )
            results.update(res)
    logger.info(results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))

    return results, coco_results, predictions
