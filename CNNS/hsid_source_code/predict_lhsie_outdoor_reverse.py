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
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#超参数定义
K = 36

from hsidataset import HsiCubicLowlightTestDataset
from model_hsid_origin import HSID_origin
from model_rdn import HSIRDN, HSIRDNDeep,HSIRDNMOD,HSIRDNECA,HSIRDNSE,HSIRDNCBAM,HSIRDNCoordAtt

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
    #print('output.shape=', output.shape)

layer_list = ["conv1",
"rdn.rdbModuleList.0",
"rdn.rdbModuleList.1",
"rdn.rdbModuleList.2",
"rdn.rdbModuleList.3",
"conv10"]

def predict_lowlight_hsid_origin():
    
    #加载模型
    #hsid = HSID(36)
    hsid = HSIRDNECA(24)
    hsid = nn.DataParallel(hsid).to(DEVICE)
    #hsid = hsid.to(DEVICE)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    save_model_path = './checkpoints/lhsie_outdoor_standard'

    hsid.load_state_dict(torch.load(save_model_path + '/hsid_rdn_eca_l1_loss_600epoch_patchsize32_best.pth', map_location='cuda:0')['gen'])

    #file_name = 'IndianPines_Data_normalized'
    file_name = 'IndianPines_Data_normalized_result'

    #加载测试数据
    batch_size = 1
    #test_data_dir = './data/test_lowlight/cuk12/'
    #test_data_dir = './data/test_lowli_outdoor_k12_indian_reversed/' + file_name + '/'
    test_data_dir = './data/test_lowli_outdoor_k12_indian/' + file_name + '/'

    test_set = HsiCubicLowlightTestDataset(test_data_dir)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    batch_size, channel, width, height = next(iter(test_dataloader))[0].shape

    band_num = len(test_dataloader)
    denoised_hsi = np.zeros((width, height, band_num))


    #指定结果输出路径
    test_result_output_path = './data/testresult/outdoor_standard_india/'
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
    for name, m in hsid.named_modules():
        print(name)
    
    print(type(hsid._modules))
    print(hsid._modules)
    #for layer_name in layer_list:
        #print('layer_name=', layer_name)
    hsid._modules.get("module")._modules.get('conv1').register_forward_hook(hook_feature)
    hsid._modules.get("module")._modules.get('rdn')._modules.get('rdbModuleList')._modules.get('0').register_forward_hook(hook_feature)
    hsid._modules.get("module")._modules.get('rdn')._modules.get('rdbModuleList')._modules.get('1').register_forward_hook(hook_feature)
    hsid._modules.get("module")._modules.get('rdn')._modules.get('rdbModuleList')._modules.get('2').register_forward_hook(hook_feature)
    hsid._modules.get("module")._modules.get('rdn')._modules.get('rdbModuleList')._modules.get('3').register_forward_hook(hook_feature)
    hsid._modules.get("module")._modules.get('conv10').register_forward_hook(hook_feature)

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
            denoised_band = noisy_test + residual
            
            denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
            denoised_band_numpy = np.squeeze(denoised_band_numpy)

            denoised_hsi[:,:,batch_idx] = denoised_band_numpy


    #mdict是python字典类型，value值需要是一个numpy数组
    scio.savemat(test_result_output_path + file_name + '_result.mat', {'denoised': denoised_hsi})

    #计算pnsr和ssim
    mpsnr = np.mean(psnr_list)
    #mssim = np.mean(ssim_list)
    #sam = SAM(denoised_hsi.transpose(2,0,1), test_label_hsi.transpose(2, 0, 1)

    for i in range(len(features_blobs)):
        print(features_blobs[i].shape)

    print('length of feature blob:', len(features_blobs))

def viz(input):
    for i in range(6):
        x = input[i+60]
        print('x.shape=', x.shape)
        #最多显示4张图
        x = x.squeeze(0)
        x = np.mean(x, axis=0)
        x = np.rot90(x,k=1, axes=(1,0))
        print('x.shape= after mean', x.shape)
        
        plt.subplot(1, 6, i+1)
        plt.axis('off')
        plt.imshow(x, cmap="gray")
    plt.show()

if __name__ == '__main__':
    predict_lowlight_hsid_origin()
    #viz(features_blobs)