import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch import nn
import os
from model import HSID,HSIDNoLocal,HSIDRes,HSIDCA,TwoStageHSID
import scipy.io as scio
import numpy as np
from hsidataset import HsiTrainDataset,HsiLowlightTestDataset
from torch.utils.data import DataLoader
import tqdm
from utils import get_adjacent_spectral_bands
from metrics import PSNR, SSIM, SAM

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#超参数定义
K = 36

from hsidataset import HsiCubicLowlightTestDataset
from model_hrnet import HyperHRNet
from model_swin_transformer import SwinIR
def predict_lowlight_residual_swim_transformer_patchsize64():
    
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

   #加载模型
    #hsid = HSID(36)
    window_size = 4
    K_adjacent_band = 36

    hsid = SwinIR(k=K_adjacent_band, upscale=1, img_size=(height, width),in_chans=1, out_chans=1,
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    #hsid = nn.DataParallel(hsid).to(DEVICE)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    hsid = nn.DataParallel(hsid).to(DEVICE)

    #hsid = hsid.to(DEVICE)
    save_model_path = './checkpoints/swinIR'
    hsid.load_state_dict(torch.load(save_model_path + '/swinIR_patchsize64_best.pth', map_location='cuda:0')['gen'])

    #指定结果输出路径
    test_result_output_path = './data/testresult/'
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
    hsid.eval()
    for batch_idx, (noisy_test, cubic_test, label_test) in enumerate(test_dataloader):
        noisy_test = noisy_test.type(torch.FloatTensor)
        label_test = label_test.type(torch.FloatTensor)
        cubic_test = cubic_test.type(torch.FloatTensor)

        noisy_test = noisy_test.to(DEVICE)
        label_test = label_test.to(DEVICE)
        cubic_test = cubic_test.to(DEVICE)

        with torch.no_grad():

            residual = hsid(noisy_test, cubic_test)
            denoised_band = noisy_test + residual
            
            denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
            denoised_band_numpy = np.squeeze(denoised_band_numpy)

            denoised_hsi[:,:,batch_idx] = denoised_band_numpy

    psnr = PSNR(denoised_hsi, test_label_hsi)
    ssim = SSIM(denoised_hsi, test_label_hsi)
    sam = SAM(denoised_hsi, test_label_hsi)

    #mdict是python字典类型，value值需要是一个numpy数组
    scio.savemat(test_result_output_path + 'result.mat', {'denoised': denoised_hsi})

    #计算pnsr和ssim
    print("=====averPSNR:{:.4f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(psnr, ssim, sam)) 

def predict_lowlight_residual_swim_transformer_patchsize20():
    
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

   #加载模型
    #hsid = HSID(36)
    window_size = 4

    hsid = SwinIR(k=36, upscale=1, img_size=(height, width),in_chans=1, out_chans=1,
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    #hsid = nn.DataParallel(hsid).to(DEVICE)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    hsid = hsid.to(DEVICE)
    save_model_path = './checkpoints/swinIR'
    hsid.load_state_dict(torch.load(save_model_path + '/swinIR_patchsize20_best.pth', map_location='cuda:0')['gen'])

    #指定结果输出路径
    test_result_output_path = './data/testresult/'
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
    hsid.eval()
    for batch_idx, (noisy_test, cubic_test, label_test) in enumerate(test_dataloader):
        noisy_test = noisy_test.type(torch.FloatTensor)
        label_test = label_test.type(torch.FloatTensor)
        cubic_test = cubic_test.type(torch.FloatTensor)

        noisy_test = noisy_test.to(DEVICE)
        label_test = label_test.to(DEVICE)
        cubic_test = cubic_test.to(DEVICE)

        with torch.no_grad():

            residual = hsid(noisy_test, cubic_test)
            denoised_band = noisy_test + residual
            
            denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
            denoised_band_numpy = np.squeeze(denoised_band_numpy)

            denoised_hsi[:,:,batch_idx] = denoised_band_numpy

    psnr = PSNR(denoised_hsi, test_label_hsi)
    ssim = SSIM(denoised_hsi, test_label_hsi)
    sam = SAM(denoised_hsi, test_label_hsi)

    #mdict是python字典类型，value值需要是一个numpy数组
    scio.savemat(test_result_output_path + 'result.mat', {'denoised': denoised_hsi})

    #计算pnsr和ssim
    print("=====averPSNR:{:.4f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(psnr, ssim, sam)) 


if __name__ == '__main__':
    #predict_lowlight_residual_swim_transformer_patchsize20()
    predict_lowlight_residual_swim_transformer_patchsize64()