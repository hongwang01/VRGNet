# This is the augmentated training phase A+. Taking baseline A=PReNet as an example
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by HongW 2020-04-11 18:10:49
#From rain generation to rain removal https://arxiv.org/abs/2008.03580
import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DerainDataset import SPATrainDataset
from SSIM import SSIM
import math
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from vrgnet import EDNet
from  derainnet import PReNet  # taking PReNet as an example
import time
from math import log,ceil

parser = argparse.ArgumentParser(description="PReNet_augtrain")
parser.add_argument("--data_path",type=str, default="./spa-data",help='path to training spa-data')
parser.add_argument('--batchSize', type=int, default=18, help='input batch size')
parser.add_argument("--niter", type=int, default=200, help="Number of training epochs")
parser.add_argument('--fake_ratio',type=float,default=0.5,help='augmentation ratio')
parser.add_argument('--patchSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='size of the RGB image')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--nef', type=int, default=32, help='channel setting for EDNet')
parser.add_argument('--resume', type=int, default=0, help='continue to train from epoch')
parser.add_argument("--milestone", type=int, default=[50,100,150], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument('--netED', default='./spamodels/ED_state_800.pt', help="trained generative model for augmetation")
parser.add_argument('--log_dir', default='./aug_spalogs/', help='tensorboard logs')
parser.add_argument('--model_dir',default='./aug_spamodels/',help='saving model')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

try:
    os.makedirs(opt.log_dir)
except OSError:
    pass
try:
    os.makedirs(opt.model_dir)
except OSError:
    pass

cudnn.benchmark = True

criterion = SSIM()
def sample_generator(netED, gt):
    random_z = torch.randn(gt.shape[0],opt.nz).cuda()
    rain_make = netED.sample(random_z) #extract G
    input_make =rain_make+gt
    return input_make

def train_model(net, netED, datasets, optimizer, lr_scheduler):
    NReal = ceil(opt.batchSize /(1+opt.fake_ratio))
    data_loader = DataLoader(datasets, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers),pin_memory=True)
    num_data = len(datasets)
    num_iter_epoch = ceil(num_data / opt.batchSize)
    writer = SummaryWriter(opt.log_dir)
    step = 0
    for epoch in range(opt.resume,opt.niter):
        mse_per_epoch = 0
        tic = time.time()
        # train stage
        lr = optimizer.param_groups[0]['lr']
        for ii, data in enumerate(data_loader):
            im_rain, im_gt = [x.cuda() for x in data]
            with torch.autograd.no_grad():
                im_rain[NReal:, ] = sample_generator(netED, im_gt[NReal:, ])
                im_rain[NReal:, ].clamp_(0.0, 1.0)
            net.train()
            optimizer.zero_grad()
            x, _ = net(im_rain)
            pixel_metric = criterion(x,im_gt)
            loss = -pixel_metric
            # back propagation
            loss.backward()
            optimizer.step()
            mse_iter = loss.item()
            mse_per_epoch += mse_iter
            if ii % 300 == 0:
                template = '[Epoch:{:>2d}/{:<2d}] {:0>5d}/{:0>5d}, Loss={:5.2e}, lr={:.2e}'
                print(template.format(epoch+1, opt.niter, ii, num_iter_epoch,mse_iter, lr))
                writer.add_scalar('Train Loss Iter', mse_iter, step)
                x1 = vutils.make_grid(x.data, normalize=True, scale_each=True)
                writer.add_image(' Derained', x1, step)
                x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                writer.add_image(' GroundTruth', x2, step)
                x3 = vutils.make_grid(im_rain, normalize=True, scale_each=True)
                writer.add_image(' Rainy Image', x3, step)
            step += 1
        mse_per_epoch /= (ii+1)
        print('Epoch:{:>2d}, Loss={:+.2e}'.format(epoch+1, mse_per_epoch))
        print('-'*100)
        # adjust the learning rate
        lr_scheduler.step()
        # save model
        save_path_model = os.path.join(opt.model_dir, 'Aug_DerainNet_state_'+str(epoch+1)+'.pt')
        torch.save(net.state_dict(), save_path_model)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc-tic))
    writer.close()
    print('Reach the maximal niter! Finish training')

if __name__ == "__main__":
    netDerain = PReNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu).cuda()  # deraining network PReNet
    netED = EDNet(opt.nc, opt.nz, opt.nef).cuda()
    netED.load_state_dict(torch.load(opt.netED))  # load trained generative model
    optimizerDerain = optim.Adam(netDerain.parameters(), lr=opt.lr)
    schedulerDerain = optim.lr_scheduler.MultiStepLR(optimizerDerain, milestones=opt.milestone, gamma=0.2)  # learning rates
    for _ in range(opt.resume):
        schedulerDerain.step()
    # optimizer
    if opt.resume:  # from opt.resume continue to train opt.resume=0 from scratch
        netDerain.load_state_dict(torch.load(os.path.join(opt.model_dir, 'Aug_DerainNet_state_' + str(opt.resume) + '.pt')))
    # training spa-data
    entire_dataset = os.path.join(opt.data_path, 'real_world.txt')
    img_files = open(entire_dataset, 'r').readlines()
    allnum_spa = len(img_files)
    len_train_dataset = ceil(3000 * (opt.fake_ratio + 1))
    train_dataset = SPATrainDataset(opt.data_path, img_files, opt.patchSize, opt.batchSize * len_train_dataset, allnum_spa)
    # train model
    train_model(netDerain, netED, train_dataset, optimizerDerain, schedulerDerain)
