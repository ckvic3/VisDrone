import json
import os

from tqdm import tqdm

from toolkit.datasets.dataset import Dataset
from toolkit.datasets.video import Video

class VisDroneVideo(Video):
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        super(VisDroneVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)



class VisDroneDataset(Dataset):
    def __init__(self,name,dataset_root,load_img=False):
        super(VisDroneDataset, self).__init__(name,dataset_root)
        with open(os.path.join("/home/tempuser1/pysot/testing_dataset/VisDrone", name+'.json'), 'r') as f:
            meta_data = json.load(f)
        pbar = tqdm(meta_data.keys(),desc="loading"+name,ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = VisDroneVideo(video,
                                             dataset_root,
                                             meta_data[video]['video_dir'],
                                             meta_data[video]['init_rect'],
                                             meta_data[video]['img_names'],
                                             meta_data[video]['gt_rect'],
                                             None)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())


if __name__ == '__main__':
    dataset_root = "/data/VisDrone Challenge/Single-Object Tracking/VisDrone2019-SOT-val/"
    dataset = VisDroneDataset("VisDrone",dataset_root)
    print(dataset.videos)