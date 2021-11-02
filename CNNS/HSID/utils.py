"""
参考的是HSID中预测的代码：
如果算法中K取24，即选择24个邻接band的话，中位数median为12
MEDIAN = 12

i表示当前通道的编号，从1到191，对于WDC这个高光谱图像BAND_NUM = 191
临接band的确定算法如下：
1，当i小于等于MEDIAN = 12时，固定取第1到第24个band（也就是前24个波段）
作为adjacent spectral bands
2，当i大于等于BAND_NUM - MEDIAN + 1，小于BAND_NUM时，
固定取BAND_NUM-K+1到BAND_NUM这24个波段（也就是最后24个波段）作为adjacent spectral bands
3，当i大于等于MEDIAN+1，小于等于BAND_NUM-K时，取i-MEDIAN到i-1(12个band) 和 
i+1到i+MEDIAN（12个band）作为adjacent spectral bands
"""
MEDIAN = 18
import numpy as np 
import torch
"""
current_band #i取值范围 从0开始到band_num-1
"""
def get_adjacent_spectral_bands(hsi_patch, K, current_band):
    batchsize, height, width, band_num = hsi_patch.shape

    output_adjacent_bands = torch.zeros((batchsize, height, width, K))
    assert(K % 2 == 0)

    MEDIAN = K//2
    i = current_band #i取值范围 从0开始到band_num-1
    if i <= MEDIAN-1:
        output_adjacent_bands = hsi_patch[:, :,:,0:K]
    elif i >= band_num - MEDIAN:
        output_adjacent_bands = hsi_patch[:, :,:, band_num-K:band_num]
    else:
        output_adjacent_bands[:, :,:, 0:MEDIAN] = hsi_patch[:, :,:, i-MEDIAN:i]
        #print(hsi_patch[:, :,:, i-MEDIAN:i].shape)
        output_adjacent_bands[:, :,:, MEDIAN:K] =  hsi_patch[:, :,:, i+1:i+MEDIAN+1]
        #print(hsi_patch[:, :,:, i+1:i+MEDIAN+1].shape)  
    return output_adjacent_bands


if __name__ == "__main__":
    hsi_patch = torch.zeros((128, 40, 20, 191))
    K = 24
    current_band = 4

    output = get_adjacent_spectral_bands(hsi_patch, K, current_band)
    print(output.shape)

    current_band = 190
    output = get_adjacent_spectral_bands(hsi_patch, K, current_band)
    print(output.shape)    

    current_band = 150
    output = get_adjacent_spectral_bands(hsi_patch, K, current_band)
    print(output.shape)  

    current_band = 12
    output = get_adjacent_spectral_bands(hsi_patch, K, current_band)
    print(output.shape) 

    current_band = 179
    output = get_adjacent_spectral_bands(hsi_patch, K, current_band)
    print(output.shape) 

    current_band = 178
    output = get_adjacent_spectral_bands(hsi_patch, K, current_band)
    print(output.shape)

    current_band = 11
    output = get_adjacent_spectral_bands(hsi_patch, K, current_band)
    print(output.shape) 