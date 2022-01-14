import torch
import os
from model import ENCAM,ENCAM_Outdoor
import scipy.io as scio
import numpy as np
from hsidataset import HsiLowlightTestDataset,HsiCubicLowlightTestDataset
from torch.utils.data import DataLoader
import tqdm
from utils import get_adjacent_spectral_bands
from metrics import PSNR, SSIM, SAM
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"


def predict_lowlight_residual():
    
    #加载模型
    encam = ENCAM_Outdoor()

    encam = encam.to(DEVICE)

    encam.eval()

    save_model_path = '../HSID/checkpoints/encam_outdoor'

    encam.load_state_dict(torch.load(save_model_path + '/encam_outdoor_lowlight_best.pth', map_location='cuda:0')['gen'])

    #加载标签数据
    #mat_src_path = '../HSID/data/lowlight_origin_outdoor_standard/test/15ms/007_2_2021-01-19_050.mat'
    #test_label_hsi = scio.loadmat(mat_src_path)['label_normalized_hsi']
    #test_label_hsi = test_label_hsi[::4,::4,::1]

    #加载测试数据
    batch_size = 1
    #test_data_dir = './data/test_lowlight/cuk12/'
    test_data_dir = '../HSID/data/test_lowli_outdoor_downsampled_k12/007_2_2021-01-20_024/'
    test_set = HsiCubicLowlightTestDataset(test_data_dir)

    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
    batch_size, channel, width, height = next(iter(test_dataloader))[0].shape
    
    band_num = len(test_dataloader)
    denoised_hsi = np.zeros((width, height, band_num))
    denoised_upsampled = np.zeros((512, 512, band_num))

    test_label_hsi = np.zeros((width, height, band_num))

    #指定结果输出路径
    test_result_output_path = './data/testresult/encam_outdoor/'
    if not os.path.exists(test_result_output_path):
        os.makedirs(test_result_output_path)



    psnr_list = []

    #逐个通道的去噪
    """
    分配一个numpy数组，存储去噪后的结果
    遍历所有通道，
    对于每个通道，通过get_adjacent_spectral_bands获取其相邻的K个通道
    调用hsid进行预测
    将预测到的residual和输入的noise加起来，得到输出band

    将去噪后的结果保存成mat结构
    """
    for batch_idx, (noisy_test, cubic_test, label_test) in enumerate(test_dataloader):
        noisy_test = noisy_test.type(torch.FloatTensor)
        label_test = label_test.type(torch.FloatTensor)
        cubic_test = cubic_test.type(torch.FloatTensor)

        noisy_test = noisy_test.to(DEVICE)
        label_test = label_test.to(DEVICE)
        cubic_test = cubic_test.to(DEVICE)

        with torch.no_grad():
            residual = encam(noisy_test, cubic_test)
            denoised_band = noisy_test + residual

            #图像上采样
            denoised_band_upsample = F.interpolate(denoised_band, scale_factor=4, mode='bilinear')
            denoised_band_up_numpy = denoised_band_upsample.cpu().numpy().astype(np.float32)
            denoised_band_up_numpy = np.squeeze(denoised_band_up_numpy)
            denoised_upsampled[:,:,batch_idx] = denoised_band_up_numpy

            denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
            denoised_band_numpy = np.squeeze(denoised_band_numpy)

            denoised_hsi[:,:,batch_idx] = denoised_band_numpy

        test_label_hsi[:,:,batch_idx] =  np.squeeze(label_test.cpu().numpy().astype(np.float32))   
        test_label_current_band = test_label_hsi[:,:,batch_idx]

        psnr = PSNR(denoised_band_numpy, test_label_current_band)
        psnr_list.append(psnr)

    mpsnr = np.mean(psnr_list)

    denoised_hsi_trans = denoised_hsi.transpose(2,0,1)
    test_label_hsi_trans = test_label_hsi.transpose(2, 0, 1)
    mssim = SSIM(denoised_hsi_trans, test_label_hsi_trans)
    sam = SAM(denoised_hsi_trans, test_label_hsi_trans)

    #计算pnsr和ssim
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(mpsnr, mssim, sam)) 

    #mdict是python字典类型，value值需要是一个numpy数组
    scio.savemat(test_result_output_path + 'result.mat', {'denoised': denoised_hsi})
    scio.savemat(test_result_output_path + 'result_upsampled.mat', {'denoised': denoised_upsampled})

if __name__=="__main__":

    predict_lowlight_residual()
