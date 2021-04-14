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
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*8, nef*16, 4, 1),
            nn.ReLU(True),
            View((-1, nef*16 * 1 * 1)),
            nn.Linear(nef*16, nz* 2),
        )
    def forward(self, input):
        distributions = self.main(input)
        return distributions

class Decoder(nn.Module):
    def __init__(self, nz, nef, nc):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, nef*16),
            View((-1, nef*16, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef*16, nef * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 8, nef * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 4, nef * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 2, nef, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef, nc, 4, 2, 1, bias=False),
            nn.ReLU(True)
        )
    def forward(self, input):
        R = self.main(input)
        return R

