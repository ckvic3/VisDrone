import imageio
import argparse
import os
import tqdm
import cv2

parser = argparse.ArgumentParser(description='make a list of image to a gif')
parser.add_argument('--path',default='sd',type=str,
        help='imgs path')
args =parser.parse_args()

def process():
    gif_imgs =[]
    # path ="/home/tempuser1/pysot/experiments/siamrpn_r50_l234_dwxcorr_gpu_visdrone/results/VisDrone/checkpoint_e20/uav0000317_02945_s/"
    for img_name in sorted(os.listdir(args.path)):
        # idx =int(img_name.replace('img','').split('.jpg')[0])
        # new_name = '{:06d}.jpg'.format(idx)
        # os.rename(os.path.join(path,img_name),os.path.join(path,new_name))
        gif_imgs.append(imageio.imread(os.path.join(args.path,img_name)))
        print(img_name)
    imageio.mimsave("test.gif",gif_imgs,fps=24)

if __name__ == '__main__':
    process()