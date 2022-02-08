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
patch_size, stride = 64, 64

k = 18
count = 0

#save_path = './data/train_lowlight/'
#save_path = './data/train_lowlight_patchsize64_train10_k12/'
#if not os.path.exists(save_path):
    #os.mkdir(save_path)
save_path_reverse = './data/train_lowlight_patchsize64_k18_d_partial/'
if not os.path.exists(save_path_reverse):
    os.mkdir(save_path_reverse)

def gen_patches(numpy_data_noisy,numpy_data_label, channel_is):
    # get multiscale patches from a single image
    channels=numpy_data_noisy.shape[0]

    print(channels)
    h, w = numpy_data_noisy.shape[1],numpy_data_noisy.shape[2]

    for channel_i in range(channel_is):
        for i in range(0, h-patch_size+1, stride):
            for j in range(0, w-patch_size+1, stride):
                noisy = numpy_data_noisy[channel_i,i:i+patch_size, j:j+patch_size]
                label = numpy_data_label[channel_i,i:i+patch_size, j:j+patch_size]
                #print(x.shape)
                if channel_i < k:
                    # print(channel_i)
                    noisy_cubic = numpy_data_noisy[0:(k*2), i:i + patch_size, j:j + patch_size]
                    # print(y.shape)
                    label_cubic = numpy_data_label[0:(k*2), i:i + patch_size, j:j + patch_size]
                elif channel_i < channels - k:
                    # print(channel_i)
                    noisy_cubic = np.concatenate((numpy_data_noisy[channel_i - k:channel_i, i:i + patch_size, j:j + patch_size],
                                        numpy_data_noisy[channel_i + 1:channel_i + k + 1, i:i + patch_size,
                                        j:j + patch_size]))
                    # print(y.shape)
                    label_cubic = np.concatenate((numpy_data_label[channel_i - k:channel_i, i:i + patch_size, j:j + patch_size],
                                        numpy_data_label[channel_i + 1:channel_i + k + 1, i:i + patch_size,
                                        j:j + patch_size]))
                else:
                    # print(channel_i)
                    noisy_cubic = numpy_data_noisy[channel_is - (k*2):channel_is, i:i + patch_size, j:j + patch_size]
                    #print(y.shape)
                    label_cubic = numpy_data_label[channel_is - (k*2):channel_is, i:i + patch_size, j:j + patch_size]
                global count
                name =  f'{count}.mat'
                file_name = save_path_reverse + name
                count = count + 1
                scio.savemat(file_name, {'patch': noisy, 'cubic': noisy_cubic, 'label': label, 'label_cubic': label_cubic})  

    #print(len(patches),len(cubic_paches))
    #return patches,cubic_paches


noisy_mat_dir = '/data2/liguanlin/codes/paper_reimplementation/CNNS/HSID/data/lowlight_origin/train/1ms'
label_mat_dir = '/data2/liguanlin/codes/paper_reimplementation/CNNS/HSID/data/lowlight_origin/train/15ms'
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

    gen_patches(noisy, label, channels)
    #gen_patches(label, noisy, channels) #for reverse dataset generation
#train=scio.loadmat('./data/origin/wdc_normalized.mat')['wdc_normalized'] #原始大小：1280*307*191