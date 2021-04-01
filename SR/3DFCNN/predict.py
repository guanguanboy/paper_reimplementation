import argparse
import h5py
import math
import numpy as np
from sklearn.metrics import  mean_squared_error
import matplotlib.pyplot as plt

import time
import scipy.io as sio
import torch
from model import ThreeDFCNN
from dataset import PaviaDataset

def predict():
    #加载待测试的数据
    f = sio.loadmat('data_process/data/pa_test.mat')
    input=f['dataa'].astype(np.float32)
    label=f['label'].astype(np.float32)
    print(input.shape) #(610, 340, 111)
    print(label.shape) #(610, 340, 111)

    test_sample = torch.randn(1, 1, 111, 33, 33)
    print(test_sample.shape)
    #模型输入shape要求：
    #mat = torch.randn(batch_size, 1, 111, 33, 33)
    #mat = torch.randn(batch_size, input_chl_num, band_num, img_width, img_height)

    #加载模型参数
    model = ThreeDFCNN()
    model.load_state_dict(torch.load("save/d3fcnn.pt"))

    #执行预测
    model.eval()

    pred = model(test_sample)
    print(pred.shape)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-M', '--model',
                        default='save/d3fcnn.h5',
                        dest='model',
                        type=str,
                        nargs=1,
                        help="The model to be used for prediction")
    option = parser.parse_args()
    predict()