# import fire
# from kitti_object_eval_python import get_label_annos
# from .eval import get_official_eval_result, get_coco_eval_result
import maskrcnn_benchmark.kitti_object_eval_python.kitti_common as kitti
from maskrcnn_benchmark.kitti_object_eval_python.eval import get_official_eval_result, get_coco_eval_result


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             coco=False,
             score_thresh=-1):
    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    # TODO need to be fix
    # new_image_annos = []
    # for anno in dt_annos:
    #     dt_anno = kitti.filter_kitti_anno(anno, ["Pedestrian", "Cyclist", "Car"])
    #     new_image_annos.append(dt_anno)
    # dt_annos = new_image_annos
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if coco:
        print(get_coco_eval_result(gt_annos, dt_annos, current_class))
    else:
        print(get_official_eval_result(gt_annos, dt_annos, current_class))


if __name__ == '__main__':
    # fire.Fire()
    label_path = "/home/jiamingsun/raid/dataset/kitti/object/training/label_2"
    result_path = "/home/abby/Repositories/maskrcnn-benchmark/output/inference/kitti_3d_val/bbox3d_result"
    label_split_file_path = "/home/jiamingsun/raid/dataset/kitti/object/ImageSets/val.txt"
    evaluate(label_path, result_path, label_split_file_path, current_class=0, coco=False, score_thresh=-1)
