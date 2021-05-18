import torch
import torch.nn as nn

from model import HSID
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from utils import get_adjacent_spectral_bands

#设置超参数
NUM_EPOCHS =100
BATCH_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INIT_LEARNING_RATE = 0.01
K = 24

def main():

    device = DEVICE
    #准备数据
    train_dataloader

    #创建模型
    net = HSID()

    #创建优化器
    optimizer = optim.Adam(net.parameters(), lr=INIT_LEARNING_RATE, betas=(0.9, 0,999))

    #定义loss 函数
    criterion = nn.MSELoss()
    for epoch in range(NUM_EPOCHS):
        
        net.train()
        for batch_idx, (lowlight, label) in enumerate(train_loader):
            lowlight = lowlight.to(device)
            label = label.to(device)

            batch_size, height, width, band_num = lowlight.shape

            """"
                our method traverses all the bands through one-by-one mode,
                which simultaneously employing spatial–spectral information
                with spatial and spatial–spectral filters, respectively
            """
            for i in range(band_num): #遍历每个band去处理
                single_lowlight_band = lowlight[:,:,:,i]
                single_label_band = label[:,:,:i]
                adj_spectral_bands = get_adjacent_spectral_bands(lowlight, K, i)

                fake_redidual = net(single_lowlight_band, adj_spectral_bands)

                true_residual = single_label_band - single_lowlight_band
                loss = criterion(true_residual, fake_redidual)

                optimizer.zero_grad()
                
                loss.backward() # calcu gradient

                optimizer.step() # update parameter

                ## Logging


if __name__ == '__main__':
    main()