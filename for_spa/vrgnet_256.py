# This is  a demo setting of EDNet  for generating patchsize = 256 x 256
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from prenet import PReNet

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size
    def forward(self, tensor):
        return tensor.view(self.size)

class Derain (nn.Module):
    def __init__(self, args):
        super(Derain, self).__init__()
        self.derainet = PReNet(args)
    def forward(self, input):
        mu_b, logvar_b = self.derainet(input)
        return mu_b, logvar_b

class EDNet(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef):
        super(EDNet,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.encoder = Encoder(self.nc,self.nef,self.nz)
        self.decoder = Decoder(self.nz,self.nef,self.nc)
    def sample (self, input):
        return  self.decoder(input)

    def forward(self, input):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        R = self.decoder(z)
        return  R, mu,logvar, z

class Encoder(nn.Module):  # RNet
    def __init__(self, nc, nef, nz):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),  # 128 x 128
            nn.ReLU(inplace=True),
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),  # 64 x 64
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),  # 32 x 32
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),  #  16 x 16
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 8, nef * 16, 4, 2, 1, bias=False),  #  8 x 8
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 16, nef * 32, 4, 2, 1, bias=False),  #  4 x 4
            # nn.BatchNorm2d(nef * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 32, nef * 64, 4, 1),  #  1 x 1
            nn.ReLU(True),
            View((-1, nef * 64 * 1 * 1)),  #
            nn.Linear(nef * 64, nz * 2),  #
        )
    def forward(self, input):
        distributions = self.main(input)
        return distributions

class Decoder(nn.Module):
    def __init__(self, nz, nef, nc):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, nef * 64),  #
            View((-1, nef * 64, 1, 1)),  # 1*1
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 64, nef * 32, 4, 1, 0, bias=False),  #  4 x 4
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 32, nef * 16, 4, 2, 1, bias=False),  #  8 x 8
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 16, nef * 8, 4, 2, 1, bias=False),  # 16 x 16
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 8, nef * 4, 4, 2, 1, bias=False),  # 32 x 32
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 4, nef * 2, 4, 2, 1, bias=False),  # 64 x 64
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 2, nef, 4, 2, 1, bias=False),  #  128 x 128
            nn.ReLU(True),
            nn.ConvTranspose2d(nef, nc, 4, 2, 1, bias=False),  # 256 x 256
            nn.ReLU(True)
        )
    def forward(self, input):
        R = self.main(input)
        return R

