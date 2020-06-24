#!/bin/bash
export PYTHONPATH=/home/tempuser1/pysot
root="/home/tempuser1/pysot"
python -u test.py --dataset VisDrone --config $root"/experiments/siamrpn_r50_l234_dwxcorr_gpu_visdrone/config.yaml" --snapshot "/home/tempuser1/pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth"