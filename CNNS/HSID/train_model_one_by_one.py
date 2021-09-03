from matplotlib.pyplot import axis, imshow
import torch
import torch.nn as nn
from torch.utils.data.dataset import ConcatDataset

from model import HSID,HSIDNoLocal,HSIDCA,HSIDRes,TwoStageHSID
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from utils import get_adjacent_spectral_bands
from hsidataset import HsiTrainDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from helper.helper_utils import init_params, get_summary_writer
import os
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.modules.loss import _Loss
from hsidataset import HsiCubicTrainDataset
import numpy as np
from metrics import PSNR, SSIM, SAM
from hsidataset import HsiCubicTestDataset,HsiCubicLowlightTestDataset
import scipy.io as scio
from losses import EdgeLoss
from tvloss import TVLoss
#from warmup_scheduler import GradualWarmupScheduler
from dir_utils import *
from model_utils import *
import time
DENOISE_PHASE = "Denoise"
ENLIGHTEN_PHASE = "Enlighten"

#设置超参数
NUM_EPOCHS = 100
BATCH_SIZE = 256
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
INIT_LEARNING_RATE = 0.001
K = 36
display_step = 20
display_band = 20
RESUME = False

#设置随机种子
seed = 200
torch.manual_seed(seed) #在CPU上设置随机种子
if DEVICE == 'cuda:1':
    torch.cuda.manual_seed(seed) #在当前GPU上设置随机种子
    torch.cuda.manual_seed_all(seed)#为所有GPU设置随机种子

recon_criterion = nn.L1Loss() 
enlighten_loss = nn.MSELoss()


from model_one_by_one import EnlightenHyperSpectralNet

def train_one_by_one_model():

    model = EnlightenHyperSpectralNet(36)

    """
    model.train_model(train_data_dir='./data/train_lowlight/',
                test_data_dir='./data/test_lowlight/cubic/',
                test_label_dir = './data/test_lowlight/origin/soup_bigcorn_orange_1ms.mat',
                batch_size=BATCH_SIZE,
                epoch_num = NUM_EPOCHS,
                init_lr = INIT_LEARNING_RATE,
                ckpt_dir = './checkpoints',
                device = DEVICE,
                display_step = display_step,
                train_phase=DENOISE_PHASE)
    """
    model.train_model(train_data_dir='./data/train_lowlight/',
                test_data_dir='./data/test_lowlight/cubic/',
                test_label_dir = './data/test_lowlight/origin/soup_bigcorn_orange_1ms.mat',
                batch_size=BATCH_SIZE,
                epoch_num = NUM_EPOCHS,
                init_lr = INIT_LEARNING_RATE,
                ckpt_dir = './checkpoints',
                device = DEVICE,
                display_step = display_step,
                train_phase=ENLIGHTEN_PHASE)

if __name__ == '__main__':
    train_one_by_one_model()