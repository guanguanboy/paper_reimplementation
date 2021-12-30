from matplotlib.pyplot import axis, imshow
import torch
import torch.nn as nn
from torch.utils.data.dataset import ConcatDataset
from datas import datagenerator
from datas import DenoisingDataset
from model_hsid_origin import HSID_origin
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from helper.helper_utils import init_params, get_summary_writer
import os
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.modules.loss import _Loss
from hsidataset import HsiCubicTrainDataset
import numpy as np
from metrics import PSNR, SSIM, SAM
from hsidataset import HsiCubicTestDataset,HsiCubicLowlightTestDataset,HsiTrainDataset
import scipy.io as scio
from losses import EdgeLoss
from tvloss import TVLoss
#from warmup_scheduler import GradualWarmupScheduler
from dir_utils import *
from model_utils import *
import time
from utils import get_adjacent_spectral_bands
from model_rdn import HSIRDN,HSIRDNMOD,HSIRDNSE,HSIRDNECA,HSIRDNWithoutECA,HSIRDNECA_Denoise
import model_utils
import dir_utils

#设置超参数
NUM_EPOCHS =100
BATCH_SIZE = 128
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
INIT_LEARNING_RATE = 0.0001
K = 36
display_step = 20
display_band = 20
RESUME = False

#设置随机种子
seed = 200
torch.manual_seed(seed) #在CPU上设置随机种子
if DEVICE == 'cuda:1':
    torch.cuda.manual_seed(seed) #在当前GPU上设置随机种子
    torch.cuda.manual_seed_all(seed)#为所有GPU设置随机种子

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

def loss_function_mse(x, y):
    MSELoss = nn.MSELoss()
    loss = MSELoss(x, y)
    return loss

recon_criterion = nn.L1Loss() 

SIGMA=100

def train_model_residual_lowlight_rdn():

    device = DEVICE
    #准备数据
    train=np.load('./data/denoise/train_washington8.npy')
    train=train.transpose((2,1,0))

    test=np.load('./data/denoise/train_washington8.npy')
    #test=test.transpose((2,1,0))
    test=test.transpose((2,1,0)) #将通道维放在最前面

    save_model_path = './checkpoints/hsirnd_denoise_l1loss_sigma100'
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    #创建模型
    net = HSIRDNECA_Denoise(K)
    init_params(net)
    #net = nn.DataParallel(net).to(device)
    net = net.to(device)

    #创建优化器
    #hsid_optimizer = optim.Adam(net.parameters(), lr=INIT_LEARNING_RATE, betas=(0.9, 0,999))
    hsid_optimizer = optim.Adam(net.parameters(), lr=INIT_LEARNING_RATE)
    scheduler = MultiStepLR(hsid_optimizer, milestones=[200,400], gamma=0.5)

    #定义loss 函数
    #criterion = nn.MSELoss()

    gen_epoch_loss_list = []

    cur_step = 0

    best_psnr = 0
    best_epoch = 0
    best_iter = 0
    start_epoch = 1
    num_epoch = 600

    mpsnr_list = []
    for epoch in range(start_epoch, num_epoch+1):
        epoch_start_time = time.time()
        scheduler.step()
        print('epoch = ', epoch, 'lr={:.6f}'.format(scheduler.get_lr()[0]))
        print(scheduler.get_lr())

        gen_epoch_loss = 0

        net.train()

        channels= 191  # 191 channels
        data_patches, data_cubic_patches = datagenerator(train, channels)

        data_patches = torch.from_numpy(data_patches.transpose((0, 3, 1, 2, )))
        data_cubic_patches = torch.from_numpy(data_cubic_patches.transpose((0, 4, 1, 2, 3)))

        DDataset = DenoisingDataset(data_patches, data_cubic_patches, SIGMA)

        print('yes')
        DLoader = DataLoader(dataset=DDataset, batch_size=BATCH_SIZE, shuffle=True)  # loader出问题了

        epoch_loss = 0
        start_time = time.time()

        #for batch_idx, (noisy, label) in enumerate([first_batch] * 300):
        for step, x_y in enumerate(DLoader):
            #print('batch_idx=', batch_idx)
            batch_x_noise, batch_y_noise, batch_x = x_y[0], x_y[1], x_y[2]
            
            batch_x_noise = batch_x_noise.to(device)
            batch_y_noise = batch_y_noise.to(device)
            batch_x = batch_x.to(device)

            hsid_optimizer.zero_grad()
            #denoised_img = net(noisy, cubic)
            #loss = loss_fuction(denoised_img, label)

            residual = net(batch_x_noise, batch_y_noise)
            alpha = 0.8
            loss = recon_criterion(residual, batch_x-batch_x_noise)
            #loss = alpha*recon_criterion(residual, label-noisy) + (1-alpha)*loss_function_mse(residual, label-noisy)
            #loss = recon_criterion(residual, label-noisy)
            loss.backward() # calcu gradient
            hsid_optimizer.step() # update parameter

            if step % 10 == 0:
                print('%4d %4d / %4d loss = %2.8f' % (
                    epoch + 1, step, data_patches.size(0) // BATCH_SIZE, loss.item() / BATCH_SIZE))

        #scheduler.step()
        #print("Decaying learning rate to %g" % scheduler.get_last_lr()[0])
 
        torch.save({
            'gen': net.state_dict(),
            'gen_opt': hsid_optimizer.state_dict(),
        }, f"{save_model_path}/hsid_rdn_eca_l1_loss_600epoch_patchsize32_{epoch}.pth")

        #测试代码
        net.eval()

        """
        channel_s = 191  # 设置多少波段
        data_patches, data_cubic_patches = datagenerator(test, channel_s)

        data_patches = torch.from_numpy(data_patches.transpose((0, 3, 1, 2,)))
        data_cubic_patches = torch.from_numpy(data_cubic_patches.transpose((0, 4, 1, 2, 3)))

        DDataset = DenoisingDataset(data_patches, data_cubic_patches, SIGMA)
        DLoader = DataLoader(dataset=DDataset, batch_size=BATCH_SIZE, shuffle=True)
        epoch_loss = 0
        
        for step, x_y in enumerate(DLoader):
            batch_x_noise, batch_y_noise, batch_x = x_y[0], x_y[1], x_y[2]

            batch_x_noise = batch_x_noise.to(DEVICE)
            batch_y_noise = batch_y_noise.to(DEVICE)
            batch_x = batch_x.to(DEVICE)
            residual = net(batch_x_noise, batch_y_noise)

            loss = loss_fuction(residual, batch_x-batch_x_noise)

            epoch_loss += loss.item()

            if step % 10 == 0:
                print('%4d %4d / %4d test loss = %2.4f' % (
                    epoch + 1, step, data_patches.size(0) // BATCH_SIZE, loss.item() / BATCH_SIZE))
        """
        #加载数据
        test_data_dir = './data/denoise/test/level100'
        test_set = HsiTrainDataset(test_data_dir)

        test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)  

        #指定结果输出路径
        test_result_output_path = './data/denoise/testresult/'
        if not os.path.exists(test_result_output_path):
            os.makedirs(test_result_output_path)

        #逐个通道的去噪
        """
        分配一个numpy数组，存储去噪后的结果
        遍历所有通道，
        对于每个通道，通过get_adjacent_spectral_bands获取其相邻的K个通道
        调用hsid进行预测
        将预测到的residual和输入的noise加起来，得到输出band

        将去噪后的结果保存成mat结构
        """
        psnr_list = []
        for batch_idx, (noisy, label) in enumerate(test_dataloader):
            noisy = noisy.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)
            
            batch_size, width, height, band_num = noisy.shape
            denoised_hsi = np.zeros((width, height, band_num))

            noisy = noisy.to(DEVICE)
            label = label.to(DEVICE)

            with torch.no_grad():
                for i in range(band_num): #遍历每个band去处理
                    current_noisy_band = noisy[:,:,:,i]
                    current_noisy_band = current_noisy_band[:,None]

                    adj_spectral_bands = get_adjacent_spectral_bands(noisy, K, i)
                    #adj_spectral_bands = torch.transpose(adj_spectral_bands,3,1) #将通道数置换到第二维  
                    adj_spectral_bands = adj_spectral_bands.permute(0, 3,1,2)                
                    adj_spectral_bands_unsqueezed = adj_spectral_bands.unsqueeze(1)
                    #print(current_noisy_band.shape, adj_spectral_bands.shape)
                    residual = net(current_noisy_band.to(DEVICE), adj_spectral_bands_unsqueezed.to(DEVICE))
                    denoised_band = residual + current_noisy_band
                    denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
                    denoised_band_numpy = np.squeeze(denoised_band_numpy)

                    denoised_hsi[:,:,i] += denoised_band_numpy

                    test_label_current_band = label[:,:,:,i]

                    label_band_numpy = test_label_current_band.cpu().numpy().astype(np.float32)
                    label_band_numpy = np.squeeze(label_band_numpy)

                    #print(denoised_band_numpy.shape, label_band_numpy.shape, label.shape)
                    psnr = PSNR(denoised_band_numpy, label_band_numpy)
                    psnr_list.append(psnr)
        
            mpsnr = np.mean(psnr_list)
            mpsnr_list.append(mpsnr)

            denoised_hsi_trans = denoised_hsi.transpose(2,0,1)
            test_label_hsi_trans = np.squeeze(label.cpu().numpy().astype(np.float32)).transpose(2, 0, 1)
            mssim = SSIM(denoised_hsi_trans, test_label_hsi_trans)
            sam = SAM(denoised_hsi_trans, test_label_hsi_trans)


            #计算pnsr和ssim
            print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(mpsnr, mssim, sam)) 

        #保存best模型
        if mpsnr > best_psnr:
            best_psnr = mpsnr
            best_epoch = epoch
            best_iter = cur_step
            torch.save({
                'epoch' : epoch,
                'gen': net.state_dict(),
                'gen_opt': hsid_optimizer.state_dict(),
            }, f"{save_model_path}/hsid_rdn_eca_l1_loss_600epoch_patchsize32_best.pth")

        print("[epoch %d it %d PSNR: %.4f --- best_epoch %d best_iter %d Best_PSNR %.4f]" % (epoch, cur_step, mpsnr, best_epoch, best_iter, best_psnr))

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, gen_epoch_loss, INIT_LEARNING_RATE))
        print("------------------------------------------------------------------")

        ##mdict是python字典类型，value值需要是一个numpy数组
        #scio.savemat(test_result_output_path + 'result.mat', {'denoised': denoised_hsi})

if __name__ == '__main__':
    train_model_residual_lowlight_rdn()