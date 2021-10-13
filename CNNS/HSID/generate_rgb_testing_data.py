
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
count = 0

#save_path = './data/test_lowlight/cubic/'
save_path = './data/test_rgb_lowlight/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
mat_src_path = './data/test_lowlight/origin/soup_bigcorn_orange_1ms.mat'

low_dir = save_path + 'low/'
if not os.path.exists(low_dir):
    os.mkdir(low_dir)

high_dir = save_path + 'high/'
if not os.path.exists(high_dir):
    os.mkdir(high_dir)

test = scio.loadmat(mat_src_path)['lowlight']
label = scio.loadmat(mat_src_path)['label']
#test=np.load('./data/origin/test_washington.npy')
test=test.transpose((2,0,1)) #将通道维放在最前面：191*1280*307
#plt.imshow(test[50,:,:])

label=label.transpose((2,0,1)) #将通道维放在最前面：191*1280*307

def gen_test_patches(numpy_data_noisy,numpy_data_label, channel_is):
    # get multiscale patches from a single image

    # get multiscale patches from a single image
    channels=numpy_data_noisy.shape[0]

    print(channels)
    h, w = numpy_data_noisy.shape[1],numpy_data_noisy.shape[2]

    for channel_i in range(channel_is):
        
        noisy = numpy_data_noisy[channel_i,:, :]
        label = numpy_data_label[channel_i,:, :]
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


channels= 64  # 191 channels

gen_test_patches(test, label, channels)