import torch
import torch.nn as nn

from model import ENCAM
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import scipy.io as scio
from helper.helper_utils import init_params, get_summary_writer
from metrics import PSNR, SSIM, SAM
from torch.utils.data import DataLoader
from hsidataset import HsiCubicTrainDataset, HsiCubicLowlightTestDataset
from torch.nn.modules.loss import _Loss
import os
import torch.nn.functional as F

#设置超参数
#设置超参数
NUM_EPOCHS =70
BATCH_SIZE = 128
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
INIT_LEARNING_RATE = 0.0001
K = 30
display_step = 20

#设置随机种子
seed = 200
torch.manual_seed(seed)
if DEVICE == 'cuda:0':
    torch.cuda.manual_seed(seed)

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

def loss_function_new(x, y):
    mseloss = nn.MSELoss()
    loss = mseloss(x, y)
    return loss

def train_model_residual_lowlight():

    device = DEVICE
    #准备数据
    train_set = HsiCubicTrainDataset('../HSID/data/train_lowlight/')
    print('total training example:', len(train_set))

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    #加载测试label数据
    mat_src_path = '../HSID/data/test_lowlight/origin/soup_bigcorn_orange_1ms.mat'
    test_label_hsi = scio.loadmat(mat_src_path)['label']

    #加载测试数据
    batch_size = 1
    test_data_dir = '../HSID/data/test_lowlight/cubic/'
    test_set = HsiCubicLowlightTestDataset(test_data_dir)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    batch_size, channel, width, height = next(iter(test_dataloader))[0].shape
    
    band_num = len(test_dataloader)
    denoised_hsi = np.zeros((width, height, band_num))

    #创建模型
    net = ENCAM()
    #init_params(net) #创建encam时，已经通过self._initialize_weights()进行了初始化
    net = net.to(device)
    #net = nn.DataParallel(net)
    #net = net.to(device)

    #创建优化器
    #hsid_optimizer = optim.Adam(net.parameters(), lr=INIT_LEARNING_RATE, betas=(0.9, 0,999))
    hsid_optimizer = optim.Adam(net.parameters(), lr=INIT_LEARNING_RATE)
    scheduler = MultiStepLR(hsid_optimizer, milestones=[15,30,45], gamma=0.1)

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

    for epoch in range(NUM_EPOCHS):
        scheduler.step()
        print(epoch, 'lr={:.6f}'.format(scheduler.get_last_lr()[0]))
        gen_epoch_loss = 0

        net.train()
        #for batch_idx, (noisy, cubic, label) in enumerate([first_batch] * 300):
        for batch_idx, (noisy, cubic, label) in enumerate(train_loader):
            #noisy, cubic, label = next(iter(train_loader)) #从dataloader中取出一个batch
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
        }, f"checkpoints/encam_{epoch}.pth")

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

                #对图像下采样
                #noisy_permute = noisy.permute(0, 3,1,2)#交换第一维和第三维 ，shape: batch_size, band_num, height, width 
                #label_permute = label.permute(0, 3, 1, 2)
                noisy_test_down = F.interpolate(noisy_test, scale_factor=0.5, mode='bilinear')
                cubic_test_squeeze = torch.squeeze(cubic_test, 0)
                cubic_test_down = F.interpolate(cubic_test_squeeze, scale_factor=0.5, mode='bilinear')
                cubic_test_down_unsqueeze = torch.unsqueeze(cubic_test_down, 0)
                residual = net(noisy_test_down, cubic_test_down_unsqueeze)
                denoised_band = noisy_test_down + residual

                #图像上采样
                denoised_band = F.interpolate(denoised_band, scale_factor=2, mode='bilinear')

                denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
                denoised_band_numpy = np.squeeze(denoised_band_numpy)

                denoised_hsi[:,:,batch_idx] = denoised_band_numpy


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
            'gen': net.state_dict(),
            'gen_opt': hsid_optimizer.state_dict(),
            }, f"checkpoints/encam_best.pth")

        print("[epoch %d it %d PSNR: %.4f --- best_epoch %d best_iter %d Best_PSNR %.4f]" % (epoch, cur_step, psnr, best_epoch, best_iter, best_psnr))
        
    tb_writer.close()

if __name__ == '__main__':
    #main()
    #train_model()
    #train_model_residual()
    train_model_residual_lowlight()