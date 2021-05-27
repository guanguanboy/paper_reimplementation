import torch
from torch.nn.modules.loss import _Loss

import numpy as np
import torch.nn.functional as F

from datas import datagenerator
from datas import DenoisingDataset
from torch.utils.data import DataLoader
import os, glob, datetime, time
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from datas import data_aug
#from models import  PNMN
import scipy.io as scio
from model import HSID
from model_origin import HSIDCNN

torch.cuda.set_device(2)
LEARNING_RATE=0.001
EPOCH=65
SIGMA=25
BATCH_SIZE=128

#train datas

#train=scio.loadmat('./data/origin/wdc_normalized.mat')['wdc_normalized'] #原始大小：1280*307*191


#train=train.transpose((2,1,0)) #将通道维放在最前面：191*1280*307

#test datas
#test=np.load('train_washington8.npy')
#test = scio.loadmat('./data/origin/GT_crop.mat')['temp']

#train datas

train=np.load('train_washington8.npy')


train=train.transpose((2,1,0))

#test datas
test=np.load('train_washington8.npy')

#test=test.transpose((2,1,0))
test=test.transpose((2,1,0)) #将通道维放在最前面
root='./'

#define denoising model
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), )
class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def loss_fuction(x,y):
    MSEloss=sum_squared_error()
    loss1=MSEloss(x,y)
    return loss1

net =  HSIDCNN()
net.cuda()
y=torch.randn(12,1,36,20,20).cuda()
x=torch.randn(12,1,20,20).cuda()
out=net(x,y)
print(out.size())

if __name__ == '__main__':
    # model selection
    print('===> Building model')
    # criterion = sum_squared_error()
    # 分别设置学习率
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE,)
    scheduler = MultiStepLR(optimizer, milestones=[15,30,45], gamma=0.25)

    for epoch in range(EPOCH):#为什么每次epoch都回增加内存占用？

        for tex in range(1):
            mode=np.random.randint(0,4)
            net.train()
            #train1=data_aug(train1,mode)
            #train2 = data_aug(train2, mode)
            #train3=data_aug(train3,mode)

            channels= 191  # 191 channels
            data_patches, data_cubic_patches = datagenerator(train, channels)

            data_patches = torch.from_numpy(data_patches.transpose((0, 3, 1, 2, )))
            data_cubic_patches = torch.from_numpy(data_cubic_patches.transpose((0, 4, 1, 2, 3)))

            DDataset = DenoisingDataset(data_patches, data_cubic_patches, SIGMA)

            print('yes')
            DLoader = DataLoader(dataset=DDataset, batch_size=BATCH_SIZE, shuffle=True)  # loader出问题了

            epoch_loss = 0
            start_time = time.time()
            for step, x_y in enumerate(DLoader):
                batch_x_noise, batch_y_noise, batch_x = x_y[0].cuda(), x_y[1].cuda(), x_y[2].cuda()

                optimizer.zero_grad()

                loss = loss_fuction(net(batch_x_noise, batch_y_noise), batch_x)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    print('%4d %4d / %4d loss = %2.4f' % (
                        epoch + 1, step, data_patches.size(0) // BATCH_SIZE, loss.item() / BATCH_SIZE))

            elapsed_time = time.time() - start_time
            log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / step, elapsed_time))

        scheduler.step()
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])

        start_time = time.time()
        net.eval()
        channel_s = 191  # 设置多少波段
        data_patches, data_cubic_patches = datagenerator(test, channel_s)

        data_patches = torch.from_numpy(data_patches.transpose((0, 3, 1, 2,)))
        data_cubic_patches = torch.from_numpy(data_cubic_patches.transpose((0, 4, 1, 2, 3)))

        DDataset = DenoisingDataset(data_patches, data_cubic_patches, SIGMA)
        DLoader = DataLoader(dataset=DDataset, batch_size=BATCH_SIZE, shuffle=True)
        epoch_loss = 0
        for step, x_y in enumerate(DLoader):
            batch_x_noise, batch_y_noise, batch_x = x_y[0].cuda(), x_y[1].cuda(), x_y[2].cuda()
            loss = loss_fuction(net(batch_x_noise, batch_y_noise), batch_x)
            epoch_loss += loss.item()

            if step % 10 == 0:
                print('%4d %4d / %4d test loss = %2.4f' % (
                    epoch + 1, step, data_patches.size(0) // BATCH_SIZE, loss.item() / BATCH_SIZE))

        elapsed_time = time.time() - start_time
        log('epcoh = %4d , test loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / step, elapsed_time))
        torch.save(net.state_dict(), 'PNMN_%03dSIGMA%03d.pth' % (epoch + 1, SIGMA))

