from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis',default=True, action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)
def main():
    # load config
    print("begin")
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)
    dataset_root = "/data/VisDrone Challenge/Single-Object Tracking/VisDrone2019-SOT-val/"
    # create model
    model = ModelBuilder()

    tracker = build_tracker(model)

    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    print(dataset.dataset_root)
    j =0
    for v_idx, video in enumerate(dataset):
        for idx, (img, gt_bbox) in enumerate(video):
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                # 左上角坐标 ，w ，h 形式
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                # 初始化tracker 的box
                img=tracker.init(img, gt_bbox_)

                img =img.cpu().numpy()[0].transpose(1,2,0)

                cv2.imwrite("./Radio{:06d}.jpg".format(j),img)
                j += 1
            else:
                break
if __name__ == '__main__':
    main()