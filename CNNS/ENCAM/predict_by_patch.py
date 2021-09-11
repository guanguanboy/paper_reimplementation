import torch
from torch import nn
import os
from model import ENCAM
import scipy.io as scio
import numpy as np
from hsidataset import HsiLowlightTestDataset
from torch.utils.data import DataLoader
import tqdm
from utils import get_adjacent_spectral_bands
from metrics import PSNR, SSIM, SAM

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

K = 30

def predict_lowlight_residual():

    #加载模型
    encam = ENCAM()
    #hsid = nn.DataParallel(hsid).to(DEVICE)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    encam = encam.to(DEVICE)

    encam.eval()
    encam.load_state_dict(torch.load('./checkpoints/encam_best_08_27.pth', map_location='cuda:0')['gen'])

    #加载数据
    mat_src_path = '../HSID/data/test_lowlight/origin/soup_bigcorn_orange_1ms.mat'
    test_label = scio.loadmat(mat_src_path)['label']
    #test=test.transpose((2,0,1)) #将通道维放在最前面：191*1280*307

    test_data_dir = '../HSID/data/test_lowlight/origin/'
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
                adj_spectral_bands = torch.unsqueeze(adj_spectral_bands, 1)
                adj_spectral_bands = adj_spectral_bands.to(DEVICE)
                print('adj_spectral_bands : ', adj_spectral_bands.shape)
                print('adj_spectral_bands shape[4] =', adj_spectral_bands.shape[4])
                #这里需要将current_noisy_band和adj_spectral_bands拆分成4份，每份大小为batchsize，1， band_num , height/2, width/2
                current_noisy_band_00 = current_noisy_band[:,:, 0:current_noisy_band.shape[2]//2, 0:current_noisy_band.shape[3]//2]
                adj_spectral_bands_00 = adj_spectral_bands[:,:,:, 0:adj_spectral_bands.shape[3]//2, 0:adj_spectral_bands.shape[4]//2]
                residual_00 = encam(current_noisy_band_00, adj_spectral_bands_00)
                denoised_band_00 = current_noisy_band_00 + residual_00

                current_noisy_band_00 = current_noisy_band[:,:, 0:current_noisy_band.shape[2]//2, 0:current_noisy_band.shape[3]//2]
                adj_spectral_bands_00 = adj_spectral_bands[:,:,:, 0:adj_spectral_bands.shape[3]//2, 0:adj_spectral_bands.shape[4]//2]
                residual_00 = encam(current_noisy_band_00, adj_spectral_bands_00)
                denoised_band_01 = current_noisy_band_00 + residual_00

                current_noisy_band_00 = current_noisy_band[:,:, 0:(current_noisy_band.shape[2]//2), 0:(current_noisy_band.shape[3]//2)]
                adj_spectral_bands_00 = adj_spectral_bands[:,:,:, 0:adj_spectral_bands.shape[3]//2, 0:adj_spectral_bands.shape[4]//2]
                residual_00 = encam(current_noisy_band_00, adj_spectral_bands_00)
                denoised_band_10 = current_noisy_band_00 + residual_00

                current_noisy_band_00 = current_noisy_band[:,:, 0:current_noisy_band.shape[2]//2, 0:current_noisy_band.shape[3]//2]
                adj_spectral_bands_11 = adj_spectral_bands[:,:,:, 0:adj_spectral_bands.shape[3]//2, 0:adj_spectral_bands.shape[4]//2]
                residual_00 = encam(current_noisy_band_00, adj_spectral_bands_00)
                denoised_band_11 = current_noisy_band_00 + residual_00

                denoised_band_0 = torch.cat((denoised_band_00,denoised_band_01), dim=3)
                denoised_band_1 = torch.cat((denoised_band_10,denoised_band_11), dim=3)
                denoised_band = torch.cat((denoised_band_0, denoised_band_1),dim=2)
                
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

    predict_lowlight_residual()
