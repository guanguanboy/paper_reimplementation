import torch
import torch.nn as nn
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
from torch.nn.modules.loss import _Loss
from hsidataset import HsiCubicTrainDataset
import numpy as np
from metrics import PSNR, SSIM, SAM
from hsidataset import HsiCubicTestDataset,HsiCubicLowlightTestDataset
import scipy.io as scio
from losses import EdgeLoss
from tvloss import TVLoss
#from warmup_scheduler import GradualWarmupScheduler
from dir_utils import *
from model_utils import *
import time

#设置超参数
NUM_EPOCHS =100
BATCH_SIZE = 256
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
INIT_LEARNING_RATE = 0.001
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

def loss_fuction_with_edge(x,y):
    MSEloss=sum_squared_error()
    loss1=MSEloss(x,y)
    edgeloss = EdgeLoss()
    loss2 = edgeloss(x, y)

    return loss1 + loss2

def loss_function_with_tvloss(x,y):
    MSEloss=sum_squared_error()
    tvloss = TVLoss()
    loss1=MSEloss(x,y)
    loss2 = tvloss(x)

    return loss1 + loss2

recon_criterion = nn.L1Loss() 


from model_multi_stage import MultiStageHSID

def train_model_multistage_lowlight():

    device = DEVICE
    #准备数据
    train_set = HsiCubicTrainDataset('./data/train_lowlight_patchsize32/')
    #print('trainset32 training example:', len(train_set32))

    #train_set_64 = HsiCubicTrainDataset('./data/train_lowlight_patchsize64/')

    #train_set_list = [train_set32, train_set_64]
    #train_set = ConcatDataset(train_set_list) #里面的样本大小必须是一致的，否则会连接失败
    print('total training example:', len(train_set))

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    #加载测试label数据
    mat_src_path = './data/test_lowlight/origin/soup_bigcorn_orange_1ms.mat'
    test_label_hsi = scio.loadmat(mat_src_path)['label']

    #加载测试数据
    batch_size = 1
    test_data_dir = './data/test_lowlight/cubic/'
    test_set = HsiCubicLowlightTestDataset(test_data_dir)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    batch_size, channel, width, height = next(iter(test_dataloader))[0].shape
    
    band_num = len(test_dataloader)
    denoised_hsi = np.zeros((width, height, band_num))

    #创建模型
    net = MultiStageHSID(K)
    init_params(net)
    #net = nn.DataParallel(net).to(device)
    net = net.to(device)

    #创建优化器
    #hsid_optimizer = optim.Adam(net.parameters(), lr=INIT_LEARNING_RATE, betas=(0.9, 0,999))
    hsid_optimizer = optim.Adam(net.parameters(), lr=INIT_LEARNING_RATE)
    scheduler = MultiStepLR(hsid_optimizer, milestones=[40,60,80], gamma=0.1)

    #定义loss 函数
    #criterion = nn.MSELoss()

    global tb_writer
    tb_writer = get_summary_writer(log_dir='logs')

    gen_epoch_loss_list = []

    cur_step = 0

    first_batch = next(iter(train_loader))

    best_psnr = 0
    best_epoch = 0
    best_iter = 0
    start_epoch = 1
    num_epoch = 100

    for epoch in range(start_epoch, num_epoch+1):
        epoch_start_time = time.time()
        scheduler.step()
        print(epoch, 'lr={:.6f}'.format(scheduler.get_last_lr()[0]))
        print(scheduler.get_lr())
        gen_epoch_loss = 0

        net.train()
        #for batch_idx, (noisy, label) in enumerate([first_batch] * 300):
        for batch_idx, (noisy, cubic, label) in enumerate(train_loader):
            #print('batch_idx=', batch_idx)
            noisy = noisy.to(device)
            label = label.to(device)
            cubic = cubic.to(device)

            hsid_optimizer.zero_grad()
            #denoised_img = net(noisy, cubic)
            #loss = loss_fuction(denoised_img, label)

            residual = net(noisy, cubic)
            loss = loss_fuction(residual, label-noisy)

            loss.backward() # calcu gradient
            hsid_optimizer.step() # update parameter

            gen_epoch_loss += loss.item()

            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Batch_idx {batch_idx}: MSE loss: {loss.item()}")
                else:
                    print("Pretrained initial state")

            tb_writer.add_scalar("MSE loss", loss.item(), cur_step)

            #step ++,每一次循环，每一个batch的处理，叫做一个step
            cur_step += 1


        gen_epoch_loss_list.append(gen_epoch_loss)
        tb_writer.add_scalar("mse epoch loss", gen_epoch_loss, epoch)

        #scheduler.step()
        #print("Decaying learning rate to %g" % scheduler.get_last_lr()[0])
 
        torch.save({
            'gen': net.state_dict(),
            'gen_opt': hsid_optimizer.state_dict(),
        }, f"checkpoints/hsid_multistage_patchsize64_{epoch}.pth")

        #测试代码
        net.eval()
        for batch_idx, (noisy_test, cubic_test, label_test) in enumerate(test_dataloader):
            noisy_test = noisy_test.type(torch.FloatTensor)
            label_test = label_test.type(torch.FloatTensor)
            cubic_test = cubic_test.type(torch.FloatTensor)

            noisy_test = noisy_test.to(DEVICE)
            label_test = label_test.to(DEVICE)
            cubic_test = cubic_test.to(DEVICE)

            with torch.no_grad():

                residual = net(noisy_test, cubic_test)
                denoised_band = noisy_test + residual
                
                denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
                denoised_band_numpy = np.squeeze(denoised_band_numpy)

                denoised_hsi[:,:,batch_idx] = denoised_band_numpy

                if batch_idx == 49:
                    residual_squeezed = torch.squeeze(residual, axis=0)
                    denoised_band_squeezed = torch.squeeze(denoised_band, axis=0) 
                    label_test_squeezed = torch.squeeze(label_test,axis=0)
                    noisy_test_squeezed = torch.squeeze(noisy_test,axis=0)
                    tb_writer.add_image(f"images/{epoch}_restored", denoised_band_squeezed, 1, dataformats='CHW')
                    tb_writer.add_image(f"images/{epoch}_residual", residual_squeezed, 1, dataformats='CHW')
                    tb_writer.add_image(f"images/{epoch}_label", label_test_squeezed, 1, dataformats='CHW')
                    tb_writer.add_image(f"images/{epoch}_noisy", noisy_test_squeezed, 1, dataformats='CHW')

        psnr = PSNR(denoised_hsi, test_label_hsi)
        ssim = SSIM(denoised_hsi, test_label_hsi)
        sam = SAM(denoised_hsi, test_label_hsi)

        #计算pnsr和ssim
        print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(psnr, ssim, sam)) 
        tb_writer.add_scalars("validation metrics", {'average PSNR':psnr,
                        'average SSIM':ssim,
                        'avarage SAM': sam}, epoch) #通过这个我就可以看到，那个epoch的性能是最好的

        #保存best模型
        if psnr > best_psnr:
            best_psnr = psnr
            best_epoch = epoch
            best_iter = cur_step
            torch.save({
                'epoch' : epoch,
                'gen': net.state_dict(),
                'gen_opt': hsid_optimizer.state_dict(),
            }, f"checkpoints/hsid_multistage_patchsize64_best.pth")

        print("[epoch %d it %d PSNR: %.4f --- best_epoch %d best_iter %d Best_PSNR %.4f]" % (epoch, cur_step, psnr, best_epoch, best_iter, best_psnr))

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, gen_epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        #保存当前模型
        torch.save({'epoch': epoch, 
                    'gen': net.state_dict(),
                    'gen_opt': hsid_optimizer.state_dict()
                    }, os.path.join('./checkpoints',"model_latest.pth"))
    tb_writer.close()


if __name__ == '__main__':

    train_model_multistage_lowlight()