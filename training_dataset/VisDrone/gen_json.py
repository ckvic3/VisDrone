from os.path import join
from os import listdir
import json
import numpy as np

VisDrone_base_path = '/home/tempuser1/VisDrone/Single-Object Tracking/'
sub_sets = ['VisDrone2019-SOT-train','VisDrone2019-SOT-val']
js = {}
for sub_set in sub_sets:
    ann_base_path = join(VisDrone_base_path, sub_set, 'annotations/')
    data_base_path =join(VisDrone_base_path, sub_set, 'sequences/')
    # n_videos =len(listdir(ann_base_path))
    for video in sorted(listdir(ann_base_path)):

        f = open(join(ann_base_path,video))
        annos = f.readlines()
        f.close()

        n_imgs = len(annos)
        video =video.strip('.txt')
        video = join(sub_set,video) # 对齐命名格式，无实际意义
        for idx,anno in enumerate(annos):
            print(video+':frame id: {:08d} / {:08d}'.format(idx+1, n_imgs))
            anno = anno.strip().split(',')
            bbox = [int(anno[0]), int(anno[1]),
                    int(anno[0]) + int(anno[2]), int(anno[1]) + int(anno[3])]
            frame = '%06d' % idx
            obj ='00'
            if video not in js:
                js[video] ={}
            if obj not in js[video]:
                js[video][obj] ={}

            js[video][obj][frame] = bbox

train = {k:v for (k,v) in js.items() if 'train' in k}
val = {k:v for (k,v) in js.items() if 'val' in k}

json.dump(train, open('train.json', 'w'), indent=4, sort_keys=True)
json.dump(val, open('val.json', 'w'), indent=4, sort_keys=True)



