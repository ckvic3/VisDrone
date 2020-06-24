from os.path import join
from os import listdir
import json

root = "../../../VisDrone/Single-Object Tracking/VisDrone2019-SOT-test-challenge/sequences"
# anno = "../../../VisDrone/Single-Object Tracking/VisDrone2019-SOT-val/annotations"
init = "../../../VisDrone/Single-Object Tracking/VisDrone2019-SOT-test-challenge/initialization"
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
    with open(join(init,path+'.txt')) as f:
        labels = f.readlines()[0]
    labels =labels.strip().split(',')
    for idx in range(len(listdir(join(root,path)))):
        if idx ==0:
            label = [int(i) for i in labels]
            js[path]["init_rect"] =label
        else:
            label=[0,0,0,0]
        gt_rects.append(label)
    js[path]["gt_rect"] =gt_rects
json.dump(js,open("VisDrone_test.json",'w'),indent=4,sort_keys=False)



