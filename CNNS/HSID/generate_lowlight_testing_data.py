
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

save_path = './data/test_lowlight/cubic/'
#save_path = './data/test_lowlight/cuk12/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
mat_src_path = './data/test_lowlight/origin/soup_bigcorn_orange_1ms.mat'

test = scio.loadmat(mat_src_path)['lowlight']
label = scio.loadmat(mat_src_path)['label']
#test=np.load('./data/origin/test_washington.npy')
test=test.transpose((2,0,1)) #将通道维放在最前面：191*1280*307
#plt.imshow(test[50,:,:])

label=label.transpose((2,0,1)) #将通道维放在最前面：191*1280*307

def gen_test_patches(numpy_data,label, channel_is):
    # get multiscale patches from a single image
    channels=numpy_data.shape[0]

    print(channels)

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
        file_name = save_path + name
        count = count + 1
        scio.savemat(file_name, {'noisy': x, 'cubic': y, 'label': x_label})  

channels= 64  # 191 channels

gen_test_patches(test, label, channels)