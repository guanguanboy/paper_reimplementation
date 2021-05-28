import torch
from torch import nn
import os
from model import HSID
import scipy.io as scio
import numpy as np
from hsidataset import HsiTrainDataset
from torch.utils.data import DataLoader
import tqdm
from utils import get_adjacent_spectral_bands
from model_origin import HSIDCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#超参数定义
K = 36

def predict():

    #加载模型
    hsid = HSIDCNN()
    #hsid = nn.DataParallel(hsid).to(DEVICE)

    hsid.load_state_dict(torch.load('./PNMN_064SIGMA025.pth'))

    #加载数据
    test_data_dir = './data/test/'
    test_set = HsiTrainDataset(test_data_dir)

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

        #noisy = noisy.to(DEVICE)
        #label = label.to(DEVICE)

        with torch.no_grad():
            for i in range(band_num): #遍历每个band去处理
                current_noisy_band = noisy[:,:,:,i]
                current_noisy_band = current_noisy_band[:,None]

                adj_spectral_bands = get_adjacent_spectral_bands(noisy, K, i)
                #adj_spectral_bands = torch.transpose(adj_spectral_bands,3,1) #将通道数置换到第二维  
                adj_spectral_bands = adj_spectral_bands.permute(0, 3,1,2)                
                adj_spectral_bands_unsqueezed = adj_spectral_bands.unsqueeze(1)

                denoised_band = hsid(current_noisy_band, adj_spectral_bands_unsqueezed)

                denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
                denoised_band_numpy = np.squeeze(denoised_band_numpy)

                denoised_hsi[:,:,i] += denoised_band_numpy

    #mdict是python字典类型，value值需要是一个numpy数组
    scio.savemat(test_result_output_path + 'result.mat', {'denoised': denoised_hsi})

    #计算pnsr和ssim





if __name__=="__main__":
    predict()