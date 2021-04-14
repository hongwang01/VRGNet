# After finishing the joint training, testing the derain module BNet
# "
import torch
import cv2
import os
import argparse
from torch.autograd import Variable
from vrgnet import Derain   # Derain: BNet,  EDNet: RNet+G
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",type=str, default="./data/rain100L/test/small/rain",help='path to testing rainy images')
parser.add_argument('--stage', type=int, default=6, help='the stage number of PReNet')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--netDerain', default='./syn100lmodels/DerainNet_state_700.pt', help="path to trained BNet")
parser.add_argument('--save_path', default='./derained_results/rain100L/', help='folder to derained images')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
try:
    os.makedirs(opt.save_path)
except OSError:
    pass

def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return False

def normalize(data):
    return data / 255.

def main():
    # Build model
    print('Loading model ...\n')
    netDerain = Derain(opt.stage).cuda()
    netDerain.load_state_dict(torch.load(opt.netDerain))
    netDerain.eval()
    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            print(img_name)
            img_path = os.path.join(opt.data_path, img_name)
            # input testing rainy image
            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])
            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y)).cuda()
            with torch.no_grad(): #
                if opt.use_gpu:
                    torch.cuda.synchronize()
                start_time = time.time()
                mu_b, logvar_b = netDerain(y)
                out = torch.clamp(mu_b, 0., 1.)
                if opt.use_gpu:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time
                print(img_name, ': ', dur_time)
                if opt.use_gpu:
                    save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
                else:
                    save_out = np.uint8(255 * out.data.numpy().squeeze())
            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])
            cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)
            count += 1
    print('Avg. time:', time_test/count)
if __name__ == "__main__":
    main()

