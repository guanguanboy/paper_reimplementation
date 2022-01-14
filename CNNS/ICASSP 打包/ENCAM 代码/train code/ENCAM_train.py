import torch
from torch.nn.modules.loss import _Loss
import numpy as np
import torch.nn.functional as F
from datas import datagenerator
from datas import DenoisingDataset
from datas import data_aug
from torch.utils.data import DataLoader
import os, glob, datetime, time
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim

from model import ENCAM
torch.cuda.set_device(0)

LEARNING_RATE=0.001
EPOCH=70

SIGMA=5 #Control the intensity of noise

BATCH_SIZE=108
#train datas
train1=np.load('cave8.npy')
train2=np.load('train_pavia8.npy')
train3=np.load('train_washington8.npy')


train1=train1.transpose((2,1,0))
train2=train2.transpose((2,1,0))
train3=train3.transpose((2,1,0))


#Verification datas
test=np.load('1.npy')
test=test[:400,:400,:]
test=test.transpose((2,1,0))
root='./'

#define MSELoss
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)
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
# load model

net =  ENCAM()
net.cuda()

# begin thr train
if __name__ == '__main__':
    print('===> Building model')
    # set the decaying learning rate
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE,)
    scheduler = MultiStepLR(optimizer, milestones=[20,35,50], gamma=0.25)

    for epoch in range(EPOCH):#
        scheduler.step(epoch)
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])
        for tex in range(1):
            mode=np.random.randint(0,4)
            net.train()
            train1=data_aug(train1,mode)
            train2 = data_aug(train2, mode)
            train3=data_aug(train3,mode)

            #Extract patches from each image
            print('epochs:', epoch)
            channels1 = 31  # 31 channels
            channels2 = 93 # 93 channels
            channels3 = 191 # 191 channels


            data_patches1, data_cubic_patches1 = datagenerator(train1, channels1)
            data_patches2, data_cubic_patches2 = datagenerator(train2, channels2)
            data_patches3, data_cubic_patches3 = datagenerator(train3, channels3)



            data_patches = np.concatenate((data_patches1, data_patches2, data_patches3,), axis=0)
            data_cubic_patches = np.concatenate((data_cubic_patches1, data_cubic_patches2, data_cubic_patches3), axis=0)

            data_patches = torch.from_numpy(data_patches.transpose((0, 3, 1, 2, )))
            data_cubic_patches = torch.from_numpy(data_cubic_patches.transpose((0, 4, 1, 2, 3)))

            DDataset = DenoisingDataset(data_patches, data_cubic_patches, SIGMA)

            print('yes')
            DLoader = DataLoader(dataset=DDataset, batch_size=BATCH_SIZE, shuffle=True)

            epoch_loss = 0
            start_time = time.time()
            #begin train
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




        #Verification process

        start_time = time.time()
        net.eval()
        channel_s = 31  # 31 channels
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
                print('%4d %4d / %4d loss = %2.4f' % (
                    epoch + 1, step, data_patches.size(0) // BATCH_SIZE, loss.item() / BATCH_SIZE))

        elapsed_time = time.time() - start_time
        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / step, elapsed_time))
        torch.save(net.state_dict(), 'ENCAM_%03dSIGMA%03d.pth' % (epoch + 1, SIGMA))

