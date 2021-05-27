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
patch_size, stride = 20, 20

k = 18
SIGMA=25

save_path = './data/train_cubic/'
def gen_patches(numpy_data,channel_is):
    # get multiscale patches from a single image
    channels=numpy_data.shape[0]

    print(channels)
    h, w = numpy_data.shape[1],numpy_data.shape[2]
    patches = []
    cubic_paches=[]

    count = 0
    for channel_i in range(channel_is):
        for i in range(0, h-patch_size+1, stride):
            for j in range(0, w-patch_size+1, stride):
                x = numpy_data[channel_i,i:i+patch_size, j:j+patch_size]
                patches.append(x)
                #print(x.shape)
                if channel_i < k:
                    # print(channel_i)
                    y = numpy_data[0:36, i:i + patch_size, j:j + patch_size]
                    # print(y.shape)
                    cubic_paches.append(y)
                elif channel_i < channels - k:
                    # print(channel_i)
                    y = np.concatenate((numpy_data[channel_i - k:channel_i, i:i + patch_size, j:j + patch_size],
                                        numpy_data[channel_i + 1:channel_i + k + 1, i:i + patch_size,
                                        j:j + patch_size]))
                    cubic_paches.append(y)
                    # print(y.shape)
                else:
                    # print(channel_i)
                    y = numpy_data[channel_is - 36:channel_is, i:i + patch_size, j:j + patch_size]
                    cubic_paches.append(y)
                    #print(y.shape)

                #给x和y分别添加噪声并且保存成mat
                x_div255=x/255.0
                y_div255=y/255.0

                print(x_div255.shape)
                #print(x_div255.size())

                print(y_div255.shape)
                #print(y_div255.size())

                x_width, x_height = x.shape
                y_chanel, y_width, y_height  = y.shape
                noise_x = np.random.randn(x_width, x_height)*(float(SIGMA) / 255.0)  # 加上高斯噪声
                noise_y= np.random.randn(y_chanel, y_width, y_height)*(float(SIGMA) / 255.0)

                batch_x_noise = x_div255 + noise_x
                batch_y_noise= y_div255+noise_y

                name =  f'{count}.mat'
                file_name = save_path + name
                count = count + 1
                scio.savemat(file_name, {'patch': batch_x_noise, 'cubic': batch_y_noise, 'label': x_div255})  

    #print(len(patches),len(cubic_paches))
    #return patches,cubic_paches



#train=scio.loadmat('./data/origin/wdc_normalized.mat')['wdc_normalized'] #原始大小：1280*307*191
train=np.load('./data/origin/train_washington8.npy')
print('before transpose shape = ', train.shape) #(1080, 307, 191)

#train=train.transpose((2,1,0)) #将通道维放在最前面,从012，变成210
#print('after transpose shape = ', train.shape) #(191, 307, 1080)

train=train.transpose((2,0,1)) #将通道维放在最前面,从012，变成210
print('after transpose shape = ', train.shape) #(191, 1080, 307)

channels= 191  # 191 channels

gen_patches(train, channels)