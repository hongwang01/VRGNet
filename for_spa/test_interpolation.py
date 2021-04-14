# "
# This is the interpolation experiment. The generator is trained on SPA-Data
# "
import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from vrgnet import EDNet  # EDNet: RNet+G
import time
import matplotlib.pyplot as plt
import random
import scipy.io as sio
from numpy.random import RandomState
from skimage import  exposure

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",type=str, default="./interpolation_results/test_data/spa-data/rain",help='path to testing rainy images a and b for latent space interpolation')
parser.add_argument("--gt_path",type=str, default="./interpolation_results/test_data/spa-data/norain",help='path to testing gt images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--nc', type=int, default=3, help='size of the RGB image')
parser.add_argument('--patch_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--stage', type=int, default=6, help='the stage number of PReNet')
parser.add_argument('--nef', type=int, default=32, help='channel setting for EDNet')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--netED', default='./spamodels/ED_state_800.pt', help="path to trained generator")
parser.add_argument('--save_patch', default='./interpolation_results/test_data/spa-data/crop_patch/', help='folder to patchs by randonmly cropping the test-data')
parser.add_argument('--save_inputfake', default='./interpolation_results/generated_data/spa-data/input_fake', help='folder to generated rainy images')
parser.add_argument('--save_rainfake', default='./interpolation_results/generated_data/spa-data/rain_fake', help='folder to generated rain layer')
opt = parser.parse_args()
if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

try:
    os.makedirs(opt.save_patch)
except OSError:
    pass

try:
    os.makedirs(opt.save_inputfake)
except OSError:
    pass
try:
    os.makedirs(opt.save_rainfake)
except OSError:
    pass

def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return False

def crop(img):
    h, w, c = img.shape
    p_h, p_w = opt.patch_size, opt.patch_size
    rand_state = RandomState(66)
    r = rand_state.randint(0, h - p_h)
    c = rand_state.randint(0, w - p_w)
    O = img[r: r + p_h, c : c + p_w]
    return O,r,c

def main():
    # Build model
    print('Loading model ...\n')
    netED = EDNet(opt.nc, opt.nz, opt.nef).cuda()
    netED.load_state_dict(torch.load(opt.netED))
    netED.eval()
    z_list=[]
    B_list=[]
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)
            gt_path = os.path.join(opt.gt_path, img_name)

            # input image
            O = cv2.imread(img_path)
            b, g, r = cv2.split(O)
            input_img = cv2.merge([r, g, b])
            # crop input to 64x64 and save
            O, row, col = crop(input_img)
            O = O.astype(np.float32) / 255
            O_patch = O
            b, g, r = cv2.split(O)
            O = cv2.merge([r, g, b])
            cv2.imwrite(os.path.join(opt.save_patch, img_name), np.uint8(255 * O))

            # crop gt image to 64*64
            B = cv2.imread(gt_path)
            b, g, r = cv2.split(B)
            gt_img = cv2.merge([r, g, b])
            B = gt_img[row: row + opt.patch_size, col: col + opt.patch_size]
            B = B.astype(np.float32) / 255
            B_patch = B

           # B_patch = B
            # save the original rain patch
            Rain_patch = O_patch-B_patch
            b, g, r = cv2.split(Rain_patch)
            Rain_patch = cv2.merge([r, g, b])
            cv2.imwrite((opt.save_patch + '/'+ 'rain'+img_name),np.uint8(255 * Rain_patch))

            # execute latent space interpolation for patch image
            O = np.expand_dims(O_patch.transpose(2, 0, 1), 0)
            O = Variable(torch.Tensor(O)).cuda()
            B = np.expand_dims(B_patch.transpose(2, 0, 1), 0)
            B = torch.Tensor(B).cuda()
            B_list.append(B)
            with torch.no_grad():  #
                if opt.use_gpu:
                    torch.cuda.synchronize()
                _, _, _,z = netED(O) # z: 1*nz
                z_list.append(z)
    for lambda_weight in (np.linspace(0,1,21)):
        z_mix = (lambda_weight * z_list[0] + (1 - lambda_weight) * z_list[1])/(np.sqrt(lambda_weight** 2+(1 - lambda_weight) **2))
        z_list.append(z_mix)
        rain_fake = netED.sample(z_mix)
        rain_fake_max = torch.max(rain_fake,1)[0]
        rain_fake = rain_fake_max.unsqueeze(dim=1).expand_as(rain_fake) #gray rain layer

        # generate fake rainy images by adding fake rain layer to different background images
        O1_fake = B_list[0] + rain_fake
        O2_fake = B_list[1] + rain_fake
        O1_fake = torch.clamp(O1_fake, 0, 1)
        O2_fake= torch.clamp(O2_fake, 0, 1)
        O1_fake = np.uint8(255 * O1_fake.data.cpu().numpy().squeeze()).transpose(1, 2, 0)
        O2_fake = np.uint8(255 * O2_fake.data.cpu().numpy().squeeze()).transpose(1, 2, 0)

        # adjust brightness and save fake rain layer
        rain_fake = torch.clamp(rain_fake, 0, 1)
        rain_fake = np.uint8(255*rain_fake.data.cpu().numpy().squeeze()).transpose(1, 2, 0)
        rain_fake = exposure.adjust_gamma(rain_fake,0.7)
        
        #save generated rainy layer and rainy images
        plt.imsave(opt.save_rainfake + '/'+str(lambda_weight)+'_'+'rainfake.png', rain_fake/255)
        plt.imsave(opt.save_inputfake + '/'+str(lambda_weight)+'_'+'O1fake.png', O1_fake/255)
        plt.imsave(opt.save_inputfake + '/'+str(lambda_weight)+'_'+'O2fake.png', O2_fake/255)
if __name__ == "__main__":
    main()

