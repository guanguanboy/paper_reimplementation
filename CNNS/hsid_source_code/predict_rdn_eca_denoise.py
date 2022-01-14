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
from model_hsid_origin import HSID_origin
from model_rdn import HSIRDN, HSIRDNDeep,HSIRDNMOD,HSIRDNECA,HSIRDNSE,HSIRDNCBAM,HSIRDNCoordAtt,HSIRDNECA_Denoise
def predict_lowlight_hsid_origin():
    
    #加载模型
    #hsid = HSID(36)
    hsid = HSIRDNECA_Denoise(K)
    hsid = nn.DataParallel(hsid).to(DEVICE)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    save_model_path = './checkpoints/hsirnd_denoise_l1loss'

    #hsid = hsid.to(DEVICE)
    hsid.load_state_dict(torch.load(save_model_path + '/hsid_rdn_eca_l1_loss_600epoch_patchsize32_best.pth', map_location='cuda:0')['gen'])

    #加载数据
    test_data_dir = './data/denoise/test/level25'
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
    hsid.eval()
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
                residual = hsid(current_noisy_band, adj_spectral_bands_unsqueezed)
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

        denoised_hsi_trans = denoised_hsi.transpose(2,0,1)
        test_label_hsi_trans = np.squeeze(label.cpu().numpy().astype(np.float32)).transpose(2, 0, 1)
        mssim = SSIM(denoised_hsi_trans, test_label_hsi_trans)
        sam = SAM(denoised_hsi_trans, test_label_hsi_trans)


        #计算pnsr和ssim
        print("=====averPSNR:{:.4f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(mpsnr, mssim, sam)) 

    #mdict是python字典类型，value值需要是一个numpy数组
    scio.savemat(test_result_output_path + 'result.mat', {'denoised': denoised_hsi})


if __name__ == '__main__':
    predict_lowlight_hsid_origin()