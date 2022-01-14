
import torch
from torch.nn.modules.loss import _Loss

import numpy as np
import torch.nn.functional as F

#from datas import datagenerator
#from datas import DenoisingDataset
from torch.utils.data import DataLoader
import os, glob, datetime, time
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
#from datas import data_aug
#from models import  PNMN
import scipy.io as scio
from model import HSID

import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from skimage import io

k = 4

#save_path = './data/test_lowlight/cubic/'
save_path = './data/test_rgb_lowlight_outdoor/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

low_dir = 'low/'
high_dir = 'high/'


def gen_test_patches(numpy_data_noisy,numpy_data_label, channel_is, mat_name):
    # get multiscale patches from a single image
    count = 0

    # get multiscale patches from a single image
    channels=numpy_data_noisy.shape[0]

    print(channels)

    file_path = os.path.join(save_path, mat_name)
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    h, w = numpy_data_noisy.shape[1],numpy_data_noisy.shape[2]

    for channel_i in range(channel_is):
        
        noisy = numpy_data_noisy[channel_i,:, :]
        label = numpy_data_label[channel_i,:, :]
        noisy_int8 = (noisy*255).astype(np.uint8)
        label_int8 = (label*255).astype(np.uint8)
        noisy_rgb = np.stack((noisy_int8, noisy_int8, noisy_int8), axis=2)
        label_rgb = np.stack((label_int8, label_int8, label_int8), axis=2)

        name =  f'{count}.png'
        count = count + 1

        noisy_file_path = os.path.join(file_path, low_dir)
        if not os.path.exists(noisy_file_path):
            os.mkdir(noisy_file_path)
        noisy_file_name = os.path.join(noisy_file_path, name)

        label_file_name = os.path.join(file_path, high_dir)
        if not os.path.exists(label_file_name):
            os.mkdir(label_file_name)
        label_file_name = os.path.join(label_file_name, name)

        io.imsave(noisy_file_name, noisy_rgb)
        io.imsave(label_file_name, label_rgb)


noisy_mat_dir = './data/lowlight_origin_outdoor_standard/test/1ms'
label_mat_dir = './data/lowlight_origin_outdoor_standard/test/15ms'
noisy_mat_list = os.listdir(noisy_mat_dir)
label_mat_list = os.listdir(label_mat_dir)
noisy_mat_list.sort()
label_mat_list.sort()

print(noisy_mat_list)
print(label_mat_list)

mat_count = len(noisy_mat_list)

channels= 64  # 191 channels


for i in range(mat_count):
    noisy = scio.loadmat(noisy_mat_dir + '/' + noisy_mat_list[i])['lowlight_normalized_hsi']
    label = scio.loadmat(label_mat_dir + '/' + label_mat_list[i])['label_normalized_hsi']

    #print(noisy.shape) #(390, 512, 64) height, width, bandnum
    #print(label.shape)
    noisy=noisy.transpose((2,0,1)) #将通道维放在最前面
    label=label.transpose((2,0,1)) #将通道维放在最前面
    mat_name = noisy_mat_list[i]
    gen_test_patches(noisy, label, channels, mat_name[:-4])# 这里的-4表示去掉.mat
