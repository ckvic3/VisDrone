from os.path import join
from os import listdir
import json

root = "../../../VisDrone/Single-Object Tracking/VisDrone2019-SOT-val/sequences"
anno = "../../../VisDrone/Single-Object Tracking/VisDrone2019-SOT-val/annotations"
js ={}
for path in sorted(listdir(root)):
    js[path]={}
    js[path]["video_dir"] = path
    js[path]["init_rect"] = []
    img_names = []
    for img_name in sorted(listdir(join(root,path))):
        img_names.append(join("sequences",path,img_name))
    js[path]["img_names"] = img_names
    gt_rects =[]
    with open(join(anno,path+'.txt')) as f:
        labels = f.readlines()
    for idx,label in enumerate(labels):
        label =label.strip().split(',')
        label =[int (i) for i in label]
        if idx==0:
            js[path]["init_rect"] =label
        gt_rects.append(label)
    js[path]["gt_rect"] =gt_rects
json.dump(js,open("VisDrone.json",'w'),indent=4,sort_keys=False)



