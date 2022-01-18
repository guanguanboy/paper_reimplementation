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
from torchstat import stat
from torchsummary import summary
from thop import profile
from thop import clever_format

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#超参数定义
K = 36

from hsidataset import HsiCubicLowlightTestDataset
from model_hsid_origin import HSID_origin, HSID_origin_3D

def predict_lowlight_hsid_origin():
    
    #加载模型
    #hsid = HSID(36)
    hsid = HSID_origin_3D(24)
    #hsid = nn.DataParallel(hsid).to(DEVICE)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    save_model_path = './checkpoints/hsid3d_origin_patchsize64/'

    hsid = hsid.to(DEVICE)
    hsid.load_state_dict(torch.load(save_model_path + '/hsid_origin_l1_loss_best.pth', map_location='cuda:0')['gen'])
    
    weight = 1280
    height = 720
    print('test hsi size: ', 'weight==', weight, 'height==', height)

    summary(hsid, input_size=[[1, weight, height], [24, weight, height]], batch_size=-1)
    
    input1 = torch.randn(1, 1, weight, height).cuda()
    input2 = torch.randn(1, 24, weight, height).cuda()

    flops, params = profile(hsid, inputs=(input1, input2))
    print(flops, params) # 1819066368.0 11689512.0
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params) # 1.819G 11.690M
    print('test hsi size: ', 'weight==', weight, 'height==', height)


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
from hsi_lptn_model import HSIRDNECA_LPTN_FUSE_CONV,HSIRDNECA_LPTN_FUSE_CONV_Without_High

def predict_lowlight_lshie_indoor_all_data():
    
    #加载模型
    #hsid = HSID(36)
    hsid = HSIRDNECA_LPTN_FUSE_CONV(24)
    #hsid = nn.DataParallel(hsid).to(DEVICE)
    hsid = hsid.to(DEVICE)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    save_model_path = './checkpoints/hsirnd_indoor_lptn_fuse_patchsize64_lr0002_lastconv'

    hsid.load_state_dict(torch.load(save_model_path + '/hsid_rdn_eca_l1_loss_600epoch_patchsize32_best.pth', map_location='cuda:0')['gen'])

    weight = 1300
    height = 1300
    print('test hsi size: ', 'weight==', weight, 'height==', height)

    summary(hsid, input_size=[[1, weight, height], [24, weight, height]], batch_size=-1)
    
    input1 = torch.randn(1, 1, weight, height).cuda()
    input2 = torch.randn(1, 24, weight, height).cuda()

    flops, params = profile(hsid, inputs=(input1, input2))
    print(flops, params) # 1819066368.0 11689512.0
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params) # 1.819G 11.690M
    print('test hsi size: ', 'weight==', weight, 'height==', height)


def predict_lowlight_lshie_without_high_indoor_all_data():
    
    #加载模型
    #hsid = HSID(36)
    hsid = HSIRDNECA_LPTN_FUSE_CONV_Without_High(24)
    #hsid = nn.DataParallel(hsid).to(DEVICE)
    hsid = hsid.to(DEVICE)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    save_model_path = './checkpoints/hsirnd_indoor_lptn_fuse_patchsize64_lr0002_lastconv'

    hsid.load_state_dict(torch.load(save_model_path + '/hsid_rdn_eca_l1_loss_600epoch_patchsize32_best.pth', map_location='cuda:0')['gen'])

    weight = 384
    height = 384
    print('test hsi size: ', 'weight==', weight, 'height==', height)

    summary(hsid, input_size=[[1, weight, height], [24, weight, height]], batch_size=-1)
    
    input1 = torch.randn(1, 1, weight, height).cuda()
    input2 = torch.randn(1, 24, weight, height).cuda()

    flops, params = profile(hsid, inputs=(input1, input2))
    print(flops, params) # 1819066368.0 11689512.0
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params) # 1.819G 11.690M
    print('test hsi size: ', 'weight==', weight, 'height==', height)


if __name__ == '__main__':
    #predict_lowlight_hsid_origin()
    #predict_lowlight_lshie_indoor_all_data()
    predict_lowlight_lshie_without_high_indoor_all_data()