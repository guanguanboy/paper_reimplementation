import torch
import os
from model import ENCAM
import scipy.io as scio
import numpy as np
from hsidataset import HsiLowlightTestDataset
from torch.utils.data import DataLoader
import tqdm
from utils import get_adjacent_spectral_bands
from metrics import PSNR, SSIM, SAM
import torch.nn.functional as F
from torchsummary import summary
from thop import profile
from thop import clever_format

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

K = 30

def predict_encam_lowlight_residual():

    #加载模型
    encam = ENCAM()
    #hsid = nn.DataParallel(hsid).to(DEVICE)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    encam = encam.to(DEVICE)

    encam.eval()
    encam.load_state_dict(torch.load('./checkpoints/encam_best.pth', map_location='cuda:0')['gen'])

    weight = 128
    height = 128
    print('test hsi size: ', 'weight==', weight, 'height==', height)

    summary(encam, input_size=[[1, weight, height], [1, 36, weight, height]], batch_size=-1)
    
    input1 = torch.randn(1, 1, weight, height).cuda()
    input2 = torch.randn(1, 1, 36, weight, height).cuda()

    flops, params = profile(encam, inputs=(input1, input2))
    print(flops, params) # 1819066368.0 11689512.0
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params) # 1.819G 11.690M
    print('test hsi size: ', 'weight==', weight, 'height==', height)    

if __name__ == '__main__':
    predict_encam_lowlight_residual()