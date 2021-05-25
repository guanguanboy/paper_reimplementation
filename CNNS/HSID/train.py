import torch
import torch.nn as nn

from model import HSID
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from utils import get_adjacent_spectral_bands
from hsidataset import HsiTrainDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from helper.helper_utils import init_params, get_summary_writer
import os
from torch.optim.lr_scheduler import MultiStepLR

#设置超参数
NUM_EPOCHS =100
BATCH_SIZE = 256
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INIT_LEARNING_RATE = 0.005
K = 24
display_step = 2
display_band = 20

#设置随机种子
seed = 200
torch.manual_seed(seed)
if DEVICE == 'cuda':
    torch.cuda.manual_seed(seed)

def main():

    device = DEVICE
    #准备数据
    train_set = HsiTrainDataset('./data/train/')
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

    #创建模型
    net = HSID()
    init_params(net)
    net = nn.DataParallel(net).to(device)

    #创建优化器
    #hsid_optimizer = optim.Adam(net.parameters(), lr=INIT_LEARNING_RATE, betas=(0.9, 0,999))
    hsid_optimizer = optim.Adam(net.parameters(), lr=INIT_LEARNING_RATE)
    scheduler = MultiStepLR(hsid_optimizer, milestones=[15,30,45], gamma=0.25)

    #定义loss 函数
    criterion = nn.MSELoss()

    global tb_writer
    tb_writer = get_summary_writer(log_dir='logs')

    gen_minibatch_loss_list = []
    gen_epoch_loss_list = []

    cur_step = 0

    first_batch = next(iter(train_loader))

    for epoch in range(NUM_EPOCHS):
        scheduler.step(epoch)
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])
        gen_epoch_loss = 0

        net.train()
        #for batch_idx, (noisy, label) in enumerate([first_batch] * 300):
        for batch_idx, (noisy, label) in enumerate(train_loader):

            noisy = noisy.to(device)
            label = label.to(device)

            batch_size, height, width, band_num = noisy.shape

            """"
                our method traverses all the bands through one-by-one mode,
                which simultaneously employing spatial–spectral information
                with spatial and spatial–spectral filters, respectively
            """
            band_loss = 0
            for i in range(band_num): #遍历每个band去处理
                single_noisy_band = noisy[:,:,:,i]
                single_noisy_band_cloned = single_noisy_band[:,None].clone()
                single_label_band = label[:,:,:,i]
                single_label_band_cloned = single_label_band[:,None].clone()

                adj_spectral_bands = get_adjacent_spectral_bands(noisy, K, i)
                #print('adj_spectral_bands.shape =', adj_spectral_bands.shape)
                #print(type(adj_spectral_bands))
                adj_spectral_bands_transposed = torch.transpose(adj_spectral_bands,3,1).clone()
                #print('transposed adj_spectral_bands.shape =', adj_spectral_bands.shape)
                #print(type(adj_spectral_bands))

                denoised_img = net(single_noisy_band_cloned, adj_spectral_bands_transposed)

                
                loss = criterion(single_label_band_cloned, denoised_img)
                
                hsid_optimizer.zero_grad()
                loss.backward() # calcu gradient
                hsid_optimizer.step() # update parameter

                ## Logging
                band_loss += loss.item()

                if i % 20 == 0:
                    print(f"Epoch {epoch}: Step {cur_step}: bandnum {i}: band MSE loss: {loss.item()}")

            gen_minibatch_loss_list.append(band_loss)
            gen_epoch_loss += band_loss

            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: MSE loss: {band_loss}")
                else:
                    print("Pretrained initial state")

            tb_writer.add_scalar("MSE loss", band_loss, cur_step)

            #step ++,每一次循环，每一个batch的处理，叫做一个step
            cur_step += 1


        gen_epoch_loss_list.append(gen_epoch_loss)
        tb_writer.add_scalar("mse epoch loss", gen_epoch_loss, epoch)

        torch.save({
            'gen': net.state_dict(),
            'gen_opt': hsid_optimizer.state_dict(),
        }, f"checkpoints/hsid_{epoch}.pth")
    tb_writer.close()



if __name__ == '__main__':
    main()