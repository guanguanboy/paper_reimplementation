import torch
import torch.nn as nn

from model import ENCAM,ENCAM_Outdoor
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
import time

#设置超参数
#设置超参数
NUM_EPOCHS =100
BATCH_SIZE = 128
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
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
    train_set = HsiCubicTrainDataset('../HSID/data/train_lowli_outdoor_standard_patchsize32_k12/')
    print('total training example:', len(train_set))

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

    #加载测试label数据
    mat_src_path = '../HSID/data/lowlight_origin_outdoor_standard/test/15ms/007_2_2021-01-20_018.mat'
    test_label_hsi = scio.loadmat(mat_src_path)['label_normalized_hsi']
    test_label_hsi = test_label_hsi[::4,::4,::1]
     #加载测试数据
    batch_size = 1
    #test_data_dir = './data/test_lowlight/cuk12/'
    test_data_dir = '../HSID/data/test_lowli_outdoor_downsampled_k12/007_2_2021-01-20_018/'

    test_set = HsiCubicLowlightTestDataset(test_data_dir)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    batch_size, channel, width, height = next(iter(test_dataloader))[0].shape
    
    band_num = len(test_dataloader)
    denoised_hsi = np.zeros((width, height, band_num))

    save_model_path = '../HSID/checkpoints/encam_outdoor2'
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)


    #创建模型
    net = ENCAM_Outdoor()
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

    mpsnr_list = []

    for epoch in range(NUM_EPOCHS):
        scheduler.step()
        print(epoch, 'lr={:.6f}'.format(scheduler.get_last_lr()[0]))
        epoch_start_time = time.time()
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
            loss = loss_fuction(noisy + residual, label)

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
            'epoch' : epoch,
            'gen': net.state_dict(),
            'gen_opt': hsid_optimizer.state_dict(),
        }, f"{save_model_path}/encam_outdoor_lowlight_{epoch}.pth")

        #测试代码
        
        net.eval()
        psnr_list = []

        for batch_idx, (noisy_test, cubic_test, label_test) in enumerate(test_dataloader):
            noisy_test = noisy_test.type(torch.FloatTensor)
            label_test = label_test.type(torch.FloatTensor)
            cubic_test = cubic_test.type(torch.FloatTensor)

            noisy_test = noisy_test.to(DEVICE)
            label_test = label_test.to(DEVICE)
            cubic_test = cubic_test.to(DEVICE)

            with torch.no_grad():

                #对图像下采样,这里做下采样的原因是，ENCAM模型训练的时候占用的GPU显存太大，导致没有办法对测试图像进行预测
                #noisy_permute = noisy.permute(0, 3,1,2)#交换第一维和第三维 ，shape: batch_size, band_num, height, width 
                #label_permute = label.permute(0, 3, 1, 2)
                #noisy_test_down = F.interpolate(noisy_test, scale_factor=0.25, mode='bilinear')
                #cubic_test_squeeze = torch.squeeze(cubic_test, 0)
                #cubic_test_down = F.interpolate(cubic_test_squeeze, scale_factor=0.25, mode='bilinear')
                #cubic_test_down_unsqueeze = torch.unsqueeze(cubic_test_down, 0)
                residual = net(noisy_test, cubic_test)
                denoised_band = noisy_test + residual

                #图像上采样
                #enoised_band = F.interpolate(denoised_band, scale_factor=4, mode='bilinear')
                
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

            test_label_current_band = test_label_hsi[:,:,batch_idx]

            psnr = PSNR(denoised_band_numpy, test_label_current_band)
            psnr_list.append(psnr)

        mpsnr = np.mean(psnr_list)
        mpsnr_list.append(mpsnr)

        denoised_hsi_trans = denoised_hsi.transpose(2,0,1)
        test_label_hsi_trans = test_label_hsi.transpose(2, 0, 1)
        mssim = SSIM(denoised_hsi_trans, test_label_hsi_trans)
        sam = SAM(denoised_hsi_trans, test_label_hsi_trans)

        #计算pnsr和ssim
        print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(mpsnr, mssim, sam)) 
        tb_writer.add_scalars("validation metrics", {'average PSNR':mpsnr,
                        'average SSIM':mssim,
                        'avarage SAM': sam}, epoch) #通过这个我就可以看到，那个epoch的性能是最好的

        #保存best模型
        if mpsnr > best_psnr:
            best_psnr = mpsnr
            best_epoch = epoch
            best_iter = cur_step
            best_ssim = mssim
            best_sam = sam
            torch.save({
                'epoch' : epoch,
                'gen': net.state_dict(),
                'gen_opt': hsid_optimizer.state_dict(),
            }, f"{save_model_path}/encam_outdoor_lowlight_best.pth")

        print("[epoch %d it %d PSNR: %.4f --- best_epoch %d best_iter %d Best_PSNR %.4f Best_SSIM %.4f Best_SAM %.4f]" % (epoch, cur_step, mpsnr, best_epoch, best_iter, best_psnr, best_ssim, best_sam))

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, gen_epoch_loss, INIT_LEARNING_RATE))
        print("------------------------------------------------------------------")

        #保存当前模型
        torch.save({'epoch': epoch, 
                    'gen': net.state_dict(),
                    'gen_opt': hsid_optimizer.state_dict(),
                    'best_psnr': best_psnr,
                    }, os.path.join(save_model_path,"model_latest.pth"))
    mpsnr_list_numpy = np.array(mpsnr_list)
    np.save(os.path.join(save_model_path, "mpsnr_per_epoch.npy"), mpsnr_list_numpy)    
    tb_writer.close()

if __name__ == '__main__':
    #main()
    #train_model()
    #train_model_residual()
    train_model_residual_lowlight()