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

#设置超参数
NUM_EPOCHS =100
BATCH_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INIT_LEARNING_RATE = 0.01
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

    #创建优化器
    #hsid_optimizer = optim.Adam(net.parameters(), lr=INIT_LEARNING_RATE, betas=(0.9, 0,999))
    hsid_optimizer = optim.Adam(net.parameters(), lr=INIT_LEARNING_RATE)
    #定义loss 函数
    criterion = nn.MSELoss()

    global tb_writer
    tb_writer = get_summary_writer(log_dir='logs')

    gen_minibatch_loss_list = []
    gen_epoch_loss_list = []

    cur_step = 0

    for epoch in range(NUM_EPOCHS):
        
        gen_epoch_loss = 0

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
            band_loss = 0
            for i in range(band_num): #遍历每个band去处理
                single_lowlight_band = lowlight[:,:,:,i]
                single_lowlight_band = single_lowlight_band[:,None]
                single_label_band = label[:,:,:,i]
                single_label_band = single_label_band[:,None]

                adj_spectral_bands = get_adjacent_spectral_bands(lowlight, K, i)
                #print('adj_spectral_bands.shape =', adj_spectral_bands.shape)
                #print(type(adj_spectral_bands))
                adj_spectral_bands = torch.transpose(adj_spectral_bands,3,1)
                #print('transposed adj_spectral_bands.shape =', adj_spectral_bands.shape)
                #print(type(adj_spectral_bands))

                fake_redidual = net(single_lowlight_band, adj_spectral_bands)

                true_residual = single_label_band - single_lowlight_band
                loss = criterion(true_residual, fake_redidual)
                
                hsid_optimizer.zero_grad()
                loss.backward() # calcu gradient
                hsid_optimizer.step() # update parameter

                ## Logging
                band_loss += loss.item()

                if i % 20 == 0:
                    print(f"Epoch {epoch}: Step {cur_step}: bandnum {i}: band MSE loss: {loss.item()}")

            gen_minibatch_loss_list.append(band_loss.item())
            gen_epoch_loss += band_loss.item()

            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: MSE loss: {band_loss.item()}")
                else:
                    print("Pretrained initial state")

            tb_writer.add_scalar("MSE loss", band_loss.item(), cur_step)

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