
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

k = 18
SIGMA=25

save_path = './data/test_cubic/'

test=np.load('./data/origin/test_washington.npy')
test=test.transpose((2,0,1)) #将通道维放在最前面：191*1280*307
plt.imshow(test[50,:,:])

def gen_test_patches(numpy_data,channel_is):
    # get multiscale patches from a single image
    channels=numpy_data.shape[0]

    print(channels)
    patches = []
    cubic_paches=[]

    count = 0
    for channel_i in range(channel_is):

        x = numpy_data[channel_i,:, :]
        patches.append(x)
        #print(x.shape)
        if channel_i < k:
            # print(channel_i)
            y = numpy_data[0:36, :, :]
            # print(y.shape)
            cubic_paches.append(y)
        elif channel_i < channels - k:
            # print(channel_i)
            y = np.concatenate((numpy_data[channel_i - k:channel_i, :, :],
                                numpy_data[channel_i + 1:channel_i + k + 1, :, :]))
            cubic_paches.append(y)
            # print(y.shape)
        else:
            # print(channel_i)
            y = numpy_data[channel_is - 36:channel_is, :, :]
            cubic_paches.append(y)
            #print(y.shape)

        #给x和y分别添加噪声并且保存成mat，由于test中的数据是已经归一化好的数据，所以这里不需要再次归一化
        #x_div255=x/255.0, 
        #y_div255=y/255.0

        x_width, x_height = x.shape
        y_chanel, y_width, y_height  = y.shape
        noise_x = np.random.randn(x_width, x_height)*(float(SIGMA) / 255.0)  # 加上高斯噪声
        noise_y= np.random.randn(y_chanel, y_width, y_height)*(float(SIGMA) / 255.0)

        batch_x_noise = x + noise_x
        batch_y_noise= y+noise_y

        name =  f'{count}.mat'
        file_name = save_path + name
        count = count + 1
        scio.savemat(file_name, {'noisy': batch_x_noise, 'cubic': batch_y_noise, 'label': x})  

channels= 191  # 191 channels

gen_test_patches(test, channels)