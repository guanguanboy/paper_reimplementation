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

from model_multi_stage import MultiStageHSID

def predict_lowlight_residual():

    #加载模型
    #hsid = HSID(36)
    hsid = MultiStageHSID(36)
    #hsid = nn.DataParallel(hsid).to(DEVICE)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    hsid = hsid.to(DEVICE)
    hsid.load_state_dict(torch.load('./checkpoints/hsid_multistage_patchsize64_best.pth', map_location='cuda:0')['gen'])

    #加载数据
    mat_src_path = './data/test_lowlight/origin/soup_bigcorn_orange_1ms.mat'
    test_label = scio.loadmat(mat_src_path)['label']
    #test=test.transpose((2,0,1)) #将通道维放在最前面：191*1280*307

    test_data_dir = './data/test_lowlight/origin/'
    test_set = HsiLowlightTestDataset(test_data_dir)

    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

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

                adj_spectral_bands = get_adjacent_spectral_bands(noisy, K, i)# shape: batch_size, width, height, band_num
                adj_spectral_bands = adj_spectral_bands.permute(0, 3,1,2)#交换第一维和第三维 ，shape: batch_size, band_num, height, width               
                adj_spectral_bands = adj_spectral_bands.to(DEVICE)
                residual = hsid(current_noisy_band, adj_spectral_bands)
                denoised_band = current_noisy_band + residual

                denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
                denoised_band_numpy = np.squeeze(denoised_band_numpy)

                denoised_hsi[:,:,i] = denoised_band_numpy

    #mdict是python字典类型，value值需要是一个numpy数组
    scio.savemat(test_result_output_path + 'result.mat', {'denoised': denoised_hsi})

    psnr = PSNR(denoised_hsi, test_label)
    ssim = SSIM(denoised_hsi, test_label)
    sam = SAM(denoised_hsi, test_label)
    #计算pnsr和ssim
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(psnr, ssim, sam)) 


if __name__=="__main__":
    #predict()
    #predict_cubic()
    #predict_residual()
    predict_lowlight_residual()