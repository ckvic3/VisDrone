# from tqdm import tqdm
# from tqdm import trange
# pbar =tqdm(total=10000000)
# pbar.set_description("processing ")
# for i in range(10000000):
#     pbar.update(1)
# pbar.close()
#
from pysot.core.config import cfg
from pysot.models.backbone import get_backbone
import torch

cfg.merge_from_file("/home/tempuser1/pysot/experiments/siamrpn_r50_l234_dwxcorr_gpu_visdrone/config.yaml")

backbone = get_backbone(cfg.BACKBONE.TYPE,
                        **cfg.BACKBONE.KWARGS)
data =torch.rand([1,3,255,255])
output = backbone.forward(data)
for i in output:
    print(i.size())