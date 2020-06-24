# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.anchor_target import AnchorTarget
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))  #当前位置
        self.name = name
        self.root = os.path.join(cur_path, '../../', root)
        self.anno = os.path.join(cur_path, '../../', anno)
        self.frame_range = frame_range  #暂时不懂有什么用
        self.num_use = num_use  #类似于限定使用多少次数据 ， -1代表每个数据只使用一次。
        self.start_idx = start_idx  #代表开始index
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            # 对标注的json进行处理
            meta_data = json.load(f)
            # 读取json数据中的轨迹信息
            meta_data = self._filter_zero(meta_data)
        # 对数据进行整理，确保格式正确
        print(len(list(meta_data.keys())),"test")
        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                # 确定命名格式为数字
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                # 按时序排序
                frames.sort()
                meta_data[video][track]['frames'] = frames
                # 删除轨迹中不包含任何内容的
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]
        # 删除video中不包含轨迹的
        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():  #对于每一个video
            new_tracks = {}
            for trk, frames in tracks.items():   #对于每一个track
                new_frames = {}
                for frm, bbox in frames.items():  #对于track中每一帧
                    if not isinstance(bbox, dict):  #判断bbox是不是一种dict类型
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        # 打乱顺序，选择不同的视频流进行训练
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        #从所有的frame中找到一个帧作为模板
        template_frame = np.random.randint(0, len(frames))
        # todo left 和right 的作用 ？？
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        # frame range 代表从前后n帧的范围内确定search frame
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        # 在之前找到的帧序列中随机确定一个作为search frame
        search_frame = np.random.choice(search_range)

        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)
    # 随机选择轨迹中的一张作为template
    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self,):
        super(TrkDataset, self).__init__()
        # 特征图尺寸判定
        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')

        # create anchor target
        self.anchor_target = AnchorTarget()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        # 使用cfg文件中的所有数据集 ，
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE, #？
                    subdata_cfg.NUM_USE,# ？
                    start
                )
            # 计算总数
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
        # 每个批次迭代多少次
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]
    #根据index确定具体的dataset
    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)
        # 按照0.5的概率进行 gray 和neg操作  Todo gray和neg的含义暂时不确定
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        if neg: # 负样本计算，即在search 中可能找不到 template
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)

        # get labels
        cls, delta, delta_weight, overlap = self.anchor_target(
                bbox, cfg.TRAIN.OUTPUT_SIZE, neg)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        return {
                'template': template,
                'search': search,
                'label_cls': cls,
                'label_loc': delta,
                'label_loc_weight': delta_weight,
                'bbox': np.array(bbox)
                }
if __name__ == '__main__':
    start = 0
    for name in cfg.DATASET.NAMES:
        subdata_cfg = getattr(cfg.DATASET, name)
        sub_dataset = SubDataset(
            name,
            subdata_cfg.ROOT,
            subdata_cfg.ANNO,
            subdata_cfg.FRAME_RANGE, #前后选取帧的范围
            subdata_cfg.NUM_USE,   #使用多少次
            start
        )
    print(len(sub_dataset))