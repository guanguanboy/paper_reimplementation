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

from hsidataset import HsiCubicLowlightTestDataset
from model_hsid_origin import HSID_origin
from model_rdn import HSIRDN, HSIRDNDeep,HSIRDNMOD,HSIRDNECA,HSIRDNSE,HSIRDNCBAM,HSIRDNCoordAtt
from hsi_lptn_model import HSIRDNECA_LPTN_FUSE_CONV

def predict_lowlight_lshie_indoor_all_data():
    
    #加载模型
    #hsid = HSID(36)
    hsid = HSIRDNECA_LPTN_FUSE_CONV(24)
    #hsid = nn.DataParallel(hsid).to(DEVICE)
    hsid = hsid.to(DEVICE)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    save_model_path = './checkpoints/hsirnd_indoor_lptn_fuse_patchsize64_lr0002_lastconv'

    hsid.load_state_dict(torch.load(save_model_path + '/hsid_rdn_eca_l1_loss_600epoch_patchsize32_best.pth', map_location='cuda:0')['gen'])

    #加载测试label数据
    mat_src_path = './data/test_lowlight/origin/soup_bigcorn_orange_1ms.mat'
    test_label_hsi = scio.loadmat(mat_src_path)['label']

    #加载Indian_pine低光照数据
    #mat_src_path = './data/indian/IndianPines_Data_normalized.mat'
    #test_label_hsi = scio.loadmat(mat_src_path)['normalized_img']

    #加载测试数据
    batch_size = 1
    test_data_dir = './data/test_lowlight/cuk12/' 
    #test_data_dir = './data/test_lowli_k12_darked_indian/IndianPines_Data_normalized_result/' 

    test_set = HsiCubicLowlightTestDataset(test_data_dir)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    batch_size, channel, width, height = next(iter(test_dataloader))[0].shape

    band_num = len(test_dataloader)
    denoised_hsi = np.zeros((width, height, band_num))


    #指定结果输出路径
    test_result_output_path = './data/testresult/lhsie_indoor/'
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

    for batch_idx, (noisy_test, cubic_test, label_test) in enumerate(test_dataloader):
        noisy_test = noisy_test.type(torch.FloatTensor)
        label_test = label_test.type(torch.FloatTensor)
        cubic_test = cubic_test.type(torch.FloatTensor)

        noisy_test = noisy_test.to(DEVICE)
        label_test = label_test.to(DEVICE)
        cubic_test = cubic_test.to(DEVICE)

        with torch.no_grad():

            residual = hsid(noisy_test, cubic_test)
            denoised_band = residual
            
            denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
            denoised_band_numpy = np.squeeze(denoised_band_numpy)

            denoised_hsi[:,:,batch_idx] = denoised_band_numpy

        test_label_current_band = test_label_hsi[:,:,batch_idx]

        psnr = PSNR(denoised_band_numpy, test_label_current_band)
        psnr_list.append(psnr)
    #mdict是python字典类型，value值需要是一个numpy数组
    scio.savemat(test_result_output_path + 'lhsie_indoor_lptn_result.mat', {'denoised': denoised_hsi})

    #计算pnsr和ssim
    mpsnr = np.mean(psnr_list)
    #mssim = np.mean(ssim_list)
    #sam = SAM(denoised_hsi.transpose(2,0,1), test_label_hsi.transpose(2, 0, 1))

    denoised_hsi_trans = denoised_hsi.transpose(2,0,1)
    test_label_hsi_trans = test_label_hsi.transpose(2, 0, 1)
    mssim = SSIM(denoised_hsi_trans, test_label_hsi_trans)
    sam = SAM(denoised_hsi_trans, test_label_hsi_trans)
    print("=====averPSNR:{:.4f}=====averSSIM:{:.4f}=====averSAM:{:.4f}".format(mpsnr, mssim, sam)) 

    #psnr_res_list = np.load(save_model_path + '/mpsnr_per_epoch.npy')
    #print(max(psnr_res_list))

if __name__ == '__main__':
    #predict_lowlight_hsid_origin()
    predict_lowlight_lshie_indoor_all_data()
