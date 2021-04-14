#  After finishing the augmentation training, testing the derain baseline. Taking PReNet+ as an example
import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
import time
from derainnet import PReNet  # taking PReNet as an example

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--data_path", type=str, default="./data/spa-data/",help='path to training data')
parser.add_argument("--model_dir", type=str, default="./aug_spamodels/Aug_DerainNet_state_200.pt", help='path to model and log files')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--save_path", type=str, default="./aug_derained_results/spa-data/", help='path to training data')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return False

def normalize(data):
    return data / 255.


try:
    os.makedirs(opt.save_path)
except OSError:
    pass

def main():
    # Build model
    print('Loading model ...\n')
    netDerain = PReNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu).cuda()  # deraining network PReNet
    netDerain.load_state_dict(torch.load(opt.model_dir))
    netDerain.eval()
    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)
            # input image
            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y))

            if opt.use_gpu:
                y = y.cuda()

            with torch.no_grad():  #
                if opt.use_gpu:
                    torch.cuda.synchronize()
                start_time = time.time()

                out, _ = netDerain(y)
                out = torch.clamp(out, 0., 1.)

                if opt.use_gpu:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(img_name, ': ', dur_time)

            if opt.use_gpu:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())  # back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)

            count += 1

    print('Avg. time:', time_test / count)


if __name__ == "__main__":
    main()

