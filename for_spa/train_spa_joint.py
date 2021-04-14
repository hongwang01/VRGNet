# This is the joint training phase
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by HongW 2020-05-11 18:10:49
"From rain generation to rain removal https://arxiv.org/abs/2008.03580"
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from vrgnet import Derain, EDNet   # Derain: BNet,  EDNet: RNet+G
from sagan_discriminator import Discriminator  # Discriminator: D
from torch.utils.data import DataLoader
from DerainDataset import SPATrainDataset
import numpy as np
from tensorboardX import SummaryWriter
import shutil
from math import log,ceil
import re
import glob
import time

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",type=str, default="./spa-data",help='path to training spa-data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=18, help='batch size')
parser.add_argument('--patchSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='size of the RGB image')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--stage', type=int, default=6, help='the stage number of PReNet')
parser.add_argument('--nef', type=int, default=32, help='channel setting for EDNet')
parser.add_argument('--ndf', type=int, default=64, help='channel setting for D')
parser.add_argument('--niter', type=int, default=800, help='the total number of training epochs')
parser.add_argument('--resume', type=int, default=0, help='continue to train from epoch')
parser.add_argument('--lambda_gp', type=float, default=10, help='penalty coefficient for wgan-gp')
parser.add_argument("--milestone", type=int, default=[400,600,650,675,690,700], help="When to decay learning rate")  #93,423,650,675,690
parser.add_argument('--lrD', type=float, default=0.0004, help='learning rate for Disciminator')
parser.add_argument('--lrDerain', type=float, default=0.0002, help='learning rate for BNet')
parser.add_argument('--lrED', type=float, default=0.0001, help='learning rate for RNet and Generator')
parser.add_argument('--n_dis', type=int, default=5, help='discriminator critic iters')
parser.add_argument('--eps2', type=float, default= 1e-6, help='prior variance for variable b')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--log_dir', default='./spalogs/', help='tensorboard logs')
parser.add_argument('--model_dir',default='./spamodels/',help='saving model')
parser.add_argument('--manualSeed', type=int, help='manual seed')
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

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'SNConv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

log_max = log(1e4)
log_min = log(1e-8)
def train_model(netDerain, netED, netD, datasets, optimizerDerain, lr_schedulerDerain, optimizerED, lr_schedulerED, optimizerD, lr_schedulerD):
    data_loader= DataLoader(datasets, batch_size=opt.batchSize,shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    num_data = len(datasets)
    num_iter_epoch = ceil(num_data / opt.batchSize)
    writer = SummaryWriter(opt.log_dir)
    step = 0
    for epoch in range(opt.resume, opt.niter):
        mse_per_epoch = 0
        tic = time.time()
        # train stage
        lrDerain = optimizerDerain.param_groups[0]['lr']
        lrED = optimizerED.param_groups[0]['lr']
        lrD = optimizerD.param_groups[0]['lr']
        print('lr_Derain %f'  % lrDerain)
        print('lr_ED %f' % lrED)
        print('lrD %f' % lrD)
        for ii, data in enumerate(data_loader):
            input, gt = [x.cuda() for x in data]
            ############################
            # (1) Update Discriminator D :
            ###########################
            # train with original data
            netD.train()
            netD.zero_grad()
            d_out_real, dr1, dr2 = netD(input)
            d_loss_real = - torch.mean(d_out_real)
            # train with fake
            mu_b, logvar_b = netDerain(input)
            rain_make, mu_z, logvar_z,_= netED(input)
            input_fake = mu_b + rain_make
            d_out_fake, df1, df2 = netD(input_fake.detach())
            d_loss_fake = d_out_fake.mean()
            # KL divergence for Gauss distribution
            logvar_b.clamp_(min=log_min, max=log_max) # clip
            var_b_div_eps = torch.div(torch.exp(logvar_b), opt.eps2)
            kl_gauss_b = 0.5 * torch.mean(
                (mu_b - gt) ** 2 / opt.eps2 + (var_b_div_eps - 1 - torch.log(var_b_div_eps)))
            logvar_z.clamp_(min=log_min, max=log_max) # clip
            var_z = torch.exp(logvar_z)
            kl_gauss_z = 0.5 * torch.mean(mu_z ** 2 + (var_z - 1 - logvar_z))
            # Compute gradient penalty
            alpha = torch.rand(input.size(0), 1, 1, 1).cuda().expand_as(input)
            interpolated = Variable(alpha * input.data + (1 - alpha) * input_fake.data, requires_grad=True)
            out, _, _ = netD(interpolated)
            grad = torch.autograd.grad(outputs=out,
                                       inputs=interpolated,
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]
            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
            # Backward + Optimize
            errD = d_loss_real + d_loss_fake + opt.lambda_gp * d_loss_gp
            errD.backward()
            optimizerD.step()
            ############################
            # (2) Update Derain network and ED network
            ###########################
            if step % opt.n_dis == 0:
                netDerain.train()
                netDerain.zero_grad()
                netED.train()
                netED.zero_grad()
                g_out_fake, _, _ = netD(input_fake)
                g_loss_fake = - g_out_fake.mean()
                errED = 0.01*g_loss_fake + kl_gauss_z + kl_gauss_b
                errED.backward()
                optimizerED.step()
                optimizerDerain.step()
            recon_loss = F.mse_loss(mu_b, gt)
            mse_iter = recon_loss.item()
            mse_per_epoch += mse_iter
            if ii % 200 == 0:
                template = '[Epoch:{:>2d}/{:<2d}] {:0>5d}/{:0>5d}, Loss={:5.2e}, gfake={:5.2e} errG={: 5.2e}'
                print(template.format(epoch+1, opt.niter, ii, num_iter_epoch, mse_iter, g_loss_fake.item(), errED.item()))
                writer.add_scalar('Derain Loss Iter', mse_iter, step)
                writer.add_scalar('Dloss', errD.item(), step)
                writer.add_scalar('EDloss', errED.item(), step)
                writer.add_scalar('drloss', d_loss_real.item(), step)
                writer.add_scalar('dfloss', d_loss_fake.item(), step)
                writer.add_scalar('gploss', opt.lambda_gp * d_loss_gp.item(), step)
                writer.add_scalar('gfloss', g_loss_fake.item(), step)
                writer.add_scalar('z_KLloss', kl_gauss_z.item(), step)
                writer.add_scalar('b_KLloss', kl_gauss_b.item(), step)
                x1 = vutils.make_grid(mu_b.data, normalize=True, scale_each=True)
                writer.add_image('Derained', x1, step)
                x2 = vutils.make_grid(gt, normalize=True, scale_each=True)
                writer.add_image('GT', x2, step)
                x3 = vutils.make_grid(input, normalize=True, scale_each=True)
                writer.add_image('Input', x3, step)
                x5 = vutils.make_grid(input_fake.data, normalize=True, scale_each=True)
                writer.add_image('Input_Fake', x5, step)
                x6 = vutils.make_grid(rain_make.data, normalize=True, scale_each=True)
                writer.add_image('Rain_fake', x6, step)
            step += 1
        mse_per_epoch /= (ii+1)
        print('Epoch:{:>2d}, Derain_Loss={:+.2e}'.format(epoch+1, mse_per_epoch))
        # adjust the learning rate
        lr_schedulerDerain.step()
        lr_schedulerED.step()
        lr_schedulerD.step()
        # save model
        save_path_model = os.path.join(opt.model_dir, 'DerainNet_state_'+str(epoch+1)+'.pt')
        torch.save(netDerain.state_dict(), save_path_model)
        save_path_model = os.path.join(opt.model_dir, 'ED_state_'+ str(epoch + 1) +'.pt')
        torch.save(netED.state_dict(), save_path_model)
        save_path_model = os.path.join(opt.model_dir, 'D_state_' + str(epoch + 1) + '.pt')
        torch.save(netD.state_dict(), save_path_model)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc-tic))
        print('-'*100)
    writer.close()
    print('Reach the maximal epochs! Finish training')

def main():
    # move the model to GPU
    netDerain = Derain(opt.stage).cuda()                     # PReNet
    netED = EDNet(opt.nc, opt.nz, opt.nef).cuda()
    netD = Discriminator(batch_size=opt.batchSize, image_size=opt.patchSize, conv_dim=opt.ndf).cuda()
    # optimizer
    optimizerDerain = optim.Adam(netDerain.parameters(), lr=opt.lrDerain)
    optimizerED = optim.Adam(netED.parameters(), lr=opt.lrED)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD)
    # scheduler
    schedulerDerain = optim.lr_scheduler.MultiStepLR(optimizerDerain, opt.milestone, gamma=0.5)
    schedulerED = optim.lr_scheduler.MultiStepLR(optimizerED, opt.milestone, gamma=0.5)
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, opt.milestone, gamma=0.5)
    # continue to train from opt.resume
    for _ in range(opt.resume):
        schedulerDerain.step()
        schedulerED.step()
        schedulerD.step()

    if opt.resume:  # from opt.resume continue to train, opt.resume=0 from scratch
        netDerain.load_state_dict(torch.load(os.path.join(opt.model_dir, 'DerainNet_state_'+ str(opt.resume)+'.pt')))
        netED.load_state_dict(torch.load(os.path.join(opt.model_dir, 'ED_state_' + str(opt.resume) + '.pt')))
        netD.load_state_dict(torch.load(os.path.join(opt.model_dir, 'D_state_' + str(opt.resume) + '.pt')))
    else:
        netDerain.apply(weights_init)
        netED.apply(weights_init)

    # training spa-data
    entire_dataset = os.path.join(opt.data_path, 'real_world.txt')
    img_files = open(entire_dataset, 'r').readlines()
    allnum_spa = len(img_files)
    train_dataset = SPATrainDataset(opt.data_path, img_files, opt.patchSize, opt.batchSize * 3000, allnum_spa)
    # train model
    train_model(netDerain, netED, netD, train_dataset, optimizerDerain, schedulerDerain, optimizerED, schedulerED, optimizerD, schedulerD)

if __name__ == '__main__':
    main()

