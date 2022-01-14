



#Fixed noise distribution for training

import numpy as np
from torch.utils.data import Dataset
import torch
patch_size, stride = 20, 20
aug_times = 1
scales = [0.5,1,1.5,2]
batch_size = 32
k=15

class DenoisingDataset(Dataset):#数据加上噪声

    def __init__(self, data_patches,data_cubic_patches, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs =data_patches
        self.cubic=data_cubic_patches
        self.sigma = sigma

    def __getitem__(self, index):
        #print(index)
        batch_x = self.xs[index]#[960,1,20,20]
        batch_y=self.cubic[index]#[960,1,24,20,20]
        batch_x=batch_x.float()/255.0
        batch_y=batch_y.float()/255.0
        noise_x = torch.randn(batch_x.size()).mul_(self.sigma/255.0)  # 加上高斯噪声
        noise_y=torch.randn(batch_y.size()).mul_(self.sigma/ 255.0)


        batch_x_noise = batch_x + noise_x
        batch_y_noise=batch_y+noise_y
        return batch_x_noise,batch_y_noise,batch_x

    def __len__(self):
        return self.xs.size(0)#??????
# tensor=torch.randn(20,1,20,20)
# tensor_2=torch.randn(20,1,1,20,20,)
#
# data88 = DenoisingDataset(tensor,tensor_2,25)
# print(data88[2][1].size())


def data_aug(img, mode=0):#图像旋转0，90，180，270，逆时针
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.rot90(img,k=1,axes=(1,2))
    elif mode == 2:
        return np.rot90(img,k=2,axes=(1,2))
    elif mode == 3:
        return np.rot90(img,k=3,axes=(1,2))







def gen_patches(numpy_data,channel_is):
    # get multiscale patches from a single image
    channels=numpy_data.shape[0]

    print(channels)
    h, w = numpy_data.shape[1],numpy_data.shape[2]
    patches = []
    cubic_paches=[]
    for channel_i in range(channel_is): #遍历band
        for i in range(0, h-patch_size+1, stride):
            for j in range(0, w-patch_size+1, stride):
                x = numpy_data[channel_i,i:i+patch_size, j:j+patch_size]
                patches.append(x)
                #print(x.shape)
                if channel_i < k:
                    # print(channel_i)
                    y = numpy_data[0:30, i:i + patch_size, j:j + patch_size]
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
                    y = numpy_data[channel_is - 30:channel_is, i:i + patch_size, j:j + patch_size]
                    cubic_paches.append(y)
                    #print(y.shape)

    #print(len(patches),len(cubic_paches))
    return patches,cubic_paches
# np.set_printoptions(threshold=np.inf)#使print大量数据不用符号...代替而显示所有
#
# dataset = gdal.Open("D:/his datasets/washington/dc.tif")
# im_width = dataset.RasterXSize #栅格矩阵的列数
# im_height = dataset.RasterYSize #栅格矩阵的行数
# im_bands = dataset.RasterCount #波段数
# im_data = dataset.ReadAsArray(0,0,im_width,im_height)#获取数据为int16数据
# uint8=im_data.astype(np.uint8)
# #print(double.shape)
# print(im_data)


def datagenerator(numpy_data,channel_is):

    # generate patches
    patches,cubic_paches= gen_patches(numpy_data,channel_is)
    #print(len(patches))
    print(len(patches),len(cubic_paches))

    data_patches = np.array(patches)
    data_cubic_patches=np.array(cubic_paches)

    print(data_patches.shape,data_cubic_patches.shape)
    data = np.expand_dims(data_patches, axis=3)
    data_cubic = np.expand_dims(data_cubic_patches, axis=4)
    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    data_cubic=np.delete(data_cubic, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data, data_cubic
