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
from skimage import io

patch_size, stride = 20, 20

k = 4
count = 0

#save_path = './data/train_lowlight/'
save_path = './data/train_lowlight_rgb/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

low_dir = save_path + 'low/'
if not os.path.exists(low_dir):
    os.mkdir(low_dir)

high_dir = save_path + 'high/'
if not os.path.exists(high_dir):
    os.mkdir(high_dir)

def gen_patches(numpy_data_noisy,numpy_data_label, channel_is):
    # get multiscale patches from a single image
    channels=numpy_data_noisy.shape[0]

    print(channels)
    h, w = numpy_data_noisy.shape[1],numpy_data_noisy.shape[2]

    for channel_i in range(channel_is):
        
        noisy = numpy_data_noisy[channel_i,0:384, 0:384]
        label = numpy_data_label[channel_i,0:384, 0:384]
        noisy_int8 = (noisy*255).astype(np.uint8)
        label_int8 = (label*255).astype(np.uint8)
        noisy_rgb = np.stack((noisy_int8, noisy_int8, noisy_int8), axis=2)
        label_rgb = np.stack((label_int8, label_int8, label_int8), axis=2)

        global count
        name =  f'{count}.png'
        noisy_file_name = low_dir + name
        label_file_name = high_dir + name
        count = count + 1

        io.imsave(noisy_file_name, noisy_rgb)
        io.imsave(label_file_name, label_rgb)

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


for i in range(5):
    noisy = scio.loadmat(noisy_mat_dir + '/' + noisy_mat_list[i])['lowlight_normalized_hsi']
    label = scio.loadmat(label_mat_dir + '/' + label_mat_list[i])['label_normalized_hsi']

    #print(noisy.shape) #(390, 512, 64) height, width, bandnum
    #print(label.shape)
    noisy=noisy.transpose((2,0,1)) #将通道维放在最前面
    label=label.transpose((2,0,1)) #将通道维放在最前面

    gen_patches(noisy, label, channels)

#train=scio.loadmat('./data/origin/wdc_normalized.mat')['wdc_normalized'] #原始大小：1280*307*191