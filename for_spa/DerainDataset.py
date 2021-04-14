import os
import os.path
import numpy as np
import torch
import cv2
import torch.utils.data as udata
from numpy.random import RandomState

class TrainDataset(udata.Dataset):
    def __init__(self, inputname, gtname,patchsize,length):
        super().__init__()
        self.patch_size=patchsize
        self.input_dir = os.path.join(inputname)
        self.gt_dir = os.path.join(gtname)
        self.img_files = os.listdir(self.input_dir)
        self.rand_state = RandomState(66)
        self.file_num = len(self.img_files)
        self.sample_num = length
    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.input_dir, file_name)
        O = cv2.imread(img_file)
        b, g, r = cv2.split(O)
        input_img = cv2.merge([r, g, b])
        O,row,col= self.crop(input_img)
        O = O.astype(np.float32) / 255
        O = np.transpose(O, (2, 0, 1))

        gt_file = os.path.join(self.gt_dir, file_name)
        B = cv2.imread(gt_file)
        b, g, r = cv2.split(B)
        gt_img = cv2.merge([r, g, b])
        B = gt_img[row: row + self.patch_size, col : col + self.patch_size]
        B = B.astype(np.float32) / 255
        B = np.transpose(B, (2, 0, 1))
        return torch.Tensor(O), torch.Tensor(B)

    def crop(self, img):
        h, w, c = img.shape
        p_h, p_w = self.patch_size, self.patch_size
        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img[r: r + p_h, c : c + p_w]
        return O,r,c

class ValDataset(udata.Dataset):
    def __init__(self, inputname,gtname):
        super().__init__()
        self.input_dir = os.path.join(inputname)
        self.gt_dir = os.path.join(gtname)
        self.img_files = os.listdir(self.input_dir)
        self.file_num = len(self.img_files)
    def __len__(self):
        return int(self.file_num)

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.input_dir, file_name)
        O = cv2.imread(img_file)
        b, g, r = cv2.split(O)
        input_img = cv2.merge([r, g, b])
        O = input_img.astype(np.float32) / 255
        O = np.transpose(O, (2, 0, 1))

        gt_file = os.path.join(self.gt_dir, file_name)
        B = cv2.imread(gt_file)
        b, g, r = cv2.split(B)
        gt_img = cv2.merge([r, g, b])
        B = gt_img.astype(np.float32) / 255
        B = np.transpose(B, (2, 0, 1))
        return torch.Tensor(O), torch.Tensor(B)

class SPATrainDataset(udata.Dataset):
    def __init__(self, dir, sub_files, patchSize,sample_num,train_num):
        super().__init__()
        self.dir = dir
        self.patch_size = patchSize
        self.sample_num= sample_num
        self.train_num = train_num
        self.sub_files=sub_files
        self.rand_state = RandomState(66)
    def __len__(self):
        return self.sample_num
    def __getitem__(self, idx):
        file_name = self.sub_files[idx % int(self.train_num)]
        input_file_name = file_name.split(' ')[0]
        gt_file_name = file_name.split(' ')[1][:-1]

        O = cv2.imread(self.dir+input_file_name)
        b, g, r = cv2.split(O)
        input_img = cv2.merge([r, g, b])
        O,row,col= self.crop(input_img)
        O = O.astype(np.float32) / 255
        O = np.transpose(O, (2, 0, 1))

        B = cv2.imread(self.dir+ gt_file_name)
        b, g, r = cv2.split(B)
        gt_img = cv2.merge([r, g, b])
        B = gt_img[row: row + self.patch_size, col : col + self.patch_size]
        B = B.astype(np.float32) / 255
        B = np.transpose(B, (2, 0, 1))
        return torch.Tensor(O), torch.Tensor(B)

    def crop(self, img):
        h, w, c = img.shape
        p_h, p_w = self.patch_size, self.patch_size
        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img[r: r + p_h, c : c + p_w]
        return O,r,c
