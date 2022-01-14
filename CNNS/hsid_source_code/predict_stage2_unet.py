import torch
from torch import nn
import os
from model_one_by_one import EnlightenHyperSpectralNet
import scipy.io as scio
import numpy as np
from torch.utils.data import DataLoader
from metrics import PSNR, SSIM, SAM
from hsidataset import HsiCubicLowlightTestDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#超参数定义
K = 36

from model_stage2_unet import HSIDDenseNetTwoStageUNet

def predict_lowlight_residual_two_stage2():

    #加载模型
    #hsid = HSID(36)
    hsid = HSIDDenseNetTwoStageUNet(36)
    #hsid = nn.DataParallel(hsid).to(DEVICE)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    hsid = hsid.to(DEVICE)
    hsid.load_state_dict(torch.load('./checkpoints/two_stage_hsid_unet_gan_best.pth', map_location='cuda:0')['gen'])

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

            residual,residual_stage2 = hsid(noisy_test, cubic_test)
            denoised_band = noisy_test + residual_stage2
            
            denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
            denoised_band_numpy = np.squeeze(denoised_band_numpy)

            denoised_hsi[:,:,batch_idx] = denoised_band_numpy

    psnr = PSNR(denoised_hsi, test_label_hsi)
    ssim = SSIM(denoised_hsi, test_label_hsi)
    sam = SAM(denoised_hsi, test_label_hsi)

    #mdict是python字典类型，value值需要是一个numpy数组
    scio.savemat(test_result_output_path + 'result.mat', {'denoised': denoised_hsi})

    #计算pnsr和ssim
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(psnr, ssim, sam)) 


if __name__ == "__main__":
    predict_lowlight_residual_two_stage2()