
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

k = 12

#save_path = './data/test_lowlight/cubic/'
#save_path = './data/test_lowli_outdoor_k12_indian_reversed/'
save_path = './data/test_lowli_k12_darked_indian/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

def gen_test_patches(numpy_data,label, channel_is, mat_name):
    # get multiscale patches from a single image
    channels=numpy_data.shape[0]

    print(channels)

    file_path = os.path.join(save_path, mat_name)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    count = 0
    for channel_i in range(channel_is):

        x = numpy_data[channel_i,:, :]
        x_label = label[channel_i,:, :]
        #print(x.shape)
        if channel_i < k:
            # print(channel_i)
            y = numpy_data[0:(k*2), :, :]
            # print(y.shape)
        elif channel_i < channels - k:
            # print(channel_i)
            y = np.concatenate((numpy_data[channel_i - k:channel_i, :, :],
                                numpy_data[channel_i + 1:channel_i + k + 1, :, :]))
            # print(y.shape)
        else:
            # print(channel_i)
            y = numpy_data[channel_is - (k*2):channel_is, :, :]
            #print(y.shape)

        name =  f'{count}.mat'
        file_name = os.path.join(file_path, name)
        count = count + 1
        scio.savemat(file_name, {'noisy': x, 'cubic': y, 'label': x_label})  

channels= 64  # 191 channels

#gen_test_patches(test, label, channels)


#noisy_mat_dir = '/mnt/liguanlin/codes/papercodes/paper_reimplementation/CNNS/HSID/data/test'
#label_mat_dir = '/mnt/liguanlin/codes/papercodes/paper_reimplementation/CNNS/HSID/data/indian'

#生成低光照的测试数据
noisy_mat_dir = '/data2/liguanlin/codes/paper_reimplementation/CNNS/HSID/data/testresult/indoor_standard_india'
label_mat_dir = '/data2/liguanlin/codes/paper_reimplementation/CNNS/HSID/data/indian'
noisy_mat_list = os.listdir(noisy_mat_dir)
label_mat_list = os.listdir(label_mat_dir)
noisy_mat_list.sort()
label_mat_list.sort()

print(noisy_mat_list)
print(label_mat_list)

mat_count = len(noisy_mat_list)

channels= 200  # 191 channels


for i in range(mat_count):
    #noisy = scio.loadmat(noisy_mat_dir + '/' + noisy_mat_list[i])['normalized_img']
    noisy = scio.loadmat(noisy_mat_dir + '/' + noisy_mat_list[i])['denoised']
    label = scio.loadmat(label_mat_dir + '/' + label_mat_list[i])['normalized_img']

    #print(noisy.shape) #(390, 512, 64) height, width, bandnum
    #print(label.shape)
    noisy=noisy.transpose((2,0,1)) #将通道维放在最前面
    label=label.transpose((2,0,1)) #将通道维放在最前面
    mat_name = noisy_mat_list[i]
    gen_test_patches(noisy, label, channels, mat_name[:-4])# 这里的-4表示去掉.mat