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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#超参数定义
K = 36

def predict():

    #加载模型
    hsid = HSID(36)
    hsid = nn.DataParallel(hsid).to(DEVICE)

    hsid.load_state_dict(torch.load('./checkpoints/hsid_5.pth')['gen'])

    #加载数据
    test=np.load('./data/origin/test_washington.npy')
    #test=test.transpose((2,0,1)) #将通道维放在最前面：191*1280*307

    test_data_dir = './data/test_level25/'
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

        noisy = noisy.to(DEVICE)
        label = label.to(DEVICE)

        with torch.no_grad():
            for i in range(band_num): #遍历每个band去处理
                current_noisy_band = noisy[:,:,:,i]
                current_noisy_band = current_noisy_band[:,None]

                adj_spectral_bands = get_adjacent_spectral_bands(noisy, K, i)# shape: batch_size, width, height, band_num
                adj_spectral_bands = torch.transpose(adj_spectral_bands,3,1)#交换第一维和第三维 ，shape: batch_size, band_num, height, width               
                denoised_band = hsid(current_noisy_band, adj_spectral_bands)

                denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
                denoised_band_numpy = np.squeeze(denoised_band_numpy)

                denoised_hsi[:,:,i] = denoised_band_numpy

    #mdict是python字典类型，value值需要是一个numpy数组
    scio.savemat(test_result_output_path + 'result.mat', {'denoised': denoised_hsi})

    psnr = PSNR(denoised_hsi, test)
    ssim = SSIM(denoised_hsi, test)
    sam = SAM(denoised_hsi, test)
    #计算pnsr和ssim
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(psnr, ssim, sam)) 

from hsidataset import HsiCubicTestDataset

def predict_cubic():
    #加载模型
    hsid = HSID(36)
    hsid = nn.DataParallel(hsid).to(DEVICE)

    hsid.load_state_dict(torch.load('./checkpoints/hsid_70.pth')['gen'])

    #加载数据
    test_label_hsi = np.load('./data/origin/test_washington.npy')

    batch_size = 1
    test_data_dir = './data/test_cubic/'
    test_set = HsiCubicTestDataset(test_data_dir)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

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
    batch_size, channel, width, height = next(iter(test_dataloader))[0].shape
    
    band_num = len(test_dataloader)
    denoised_hsi = np.zeros((width, height, band_num))

    for batch_idx, (noisy, cubic, label) in enumerate(test_dataloader):
        noisy = noisy.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        cubic = cubic.type(torch.FloatTensor)

        batch_size, width, height, band_num = noisy.shape

        noisy = noisy.to(DEVICE)
        cubic = cubic.to(DEVICE)

        with torch.no_grad():
                       
            denoised_band = hsid(noisy, cubic)

            denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
            denoised_band_numpy = np.squeeze(denoised_band_numpy)

            denoised_hsi[:,:,batch_idx] = denoised_band_numpy

    #mdict是python字典类型，value值需要是一个numpy数组
    scio.savemat(test_result_output_path + 'result.mat', {'denoised': denoised_hsi})

    psnr = PSNR(denoised_hsi, test_label_hsi)
    ssim = SSIM(denoised_hsi, test_label_hsi)
    sam = SAM(denoised_hsi, test_label_hsi)

    #计算pnsr和ssim
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(psnr, ssim, sam)) 
  

def predict_residual():

    #加载模型
    hsid = HSID(36)
    #hsid = nn.DataParallel(hsid).to(DEVICE)
    hsid = hsid.to(DEVICE)
    hsid.load_state_dict(torch.load('./checkpoints/hsid_99.pth')['gen'])

    #加载数据
    test=np.load('./data/origin/test_washington.npy')
    #test=test.transpose((2,0,1)) #将通道维放在最前面：191*1280*307

    test_data_dir = './data/test_level25/'
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

    psnr = PSNR(denoised_hsi, test)
    ssim = SSIM(denoised_hsi, test)
    sam = SAM(denoised_hsi, test)
    #计算pnsr和ssim
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(psnr, ssim, sam)) 

def predict_lowlight_residual():

    #加载模型
    #hsid = HSID(36)
    hsid = HSID(36)
    #hsid = nn.DataParallel(hsid).to(DEVICE)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    hsid = hsid.to(DEVICE)
    hsid.load_state_dict(torch.load('./checkpoints/hsid_99.pth', map_location='cuda:0')['gen'])

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

from model_res import HSIDRefactoredTwoStage

def predict_lowlight_residual_two_stage():

    #加载模型
    #hsid = HSID(36)
    hsid = HSIDRefactoredTwoStage(36)
    hsid = nn.DataParallel(hsid).to(DEVICE) #如果是使用多GPU并行训练出来的，那么在预测时，也需要使用DataParallel将网络包起来
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #hsid = hsid.to(DEVICE)
    hsid.load_state_dict(torch.load('./checkpoints/two_stage_hsid_resnet_199.pth', map_location='cuda:0')['gen'])

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
    hsid.eval()
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
                residual,residual_stage2 = hsid(current_noisy_band, adj_spectral_bands)
                denoised_band = current_noisy_band + residual_stage2

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
    #predict_lowlight_residual()
    predict_lowlight_residual_two_stage()