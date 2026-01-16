import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from net.selfSetNet02 import SelfSetNet
from utils.tdataloader import test_dataset
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size')

parser.add_argument('--pth_path', type=str, default='./checkpoints/CamoMS/CamoMS.pth')

opt = parser.parse_args()
model = SelfSetNet()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

print('start test...')
for _data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:
    t1 = time.time()
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/CamoMS/{}/'.format(_data_name)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+'edge/', exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        pmaps, masks = model(image)
        res = pmaps[0]
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path+name, (res*255).astype(np.uint8))
    t2 = time.time()
    print("The total time for ", _data_name, " is ", t2-t1, " seconds.")

print("test finished...")
