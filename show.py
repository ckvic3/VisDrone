from os.path import join, isdir
from os import listdir, mkdir, makedirs
import os
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
from concurrent import futures
import sys
import time
VisDrone_base_path = '/home/tempuser1/VisDrone/Single-Object Tracking/train/VisDrone2019-SOT-train'
save ='/home/tempuser1/save/'
if not isdir(save): makedirs(save)
ann_base_path = join(VisDrone_base_path, 'annotations/')
data_base_path =join(VisDrone_base_path, 'sequences/')
for video in listdir(ann_base_path):
    FramePaths = sorted(listdir(join(data_base_path,str(video).strip('.txt'))))
    f =open(join(ann_base_path,video))
    annos = f.readlines()
    f.close()
    for idx,FramePath in enumerate(FramePaths):
        im = cv2.imread(join(data_base_path,str(video).strip('.txt'),FramePath))
        avg_chans = np.mean(im, axis=(0, 1))
        anno =annos[idx].strip().split(',')

        bbox =[int(anno[0]), int(anno[1]),
               int(anno[0])+int(anno[2]),int(anno[1])+int(anno[3]) ]
        im=cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color=(255,0,0),thickness=1)
        cv2.imwrite(join(save,'{:06d}.jpg'.format(idx)),im)
    exit()

