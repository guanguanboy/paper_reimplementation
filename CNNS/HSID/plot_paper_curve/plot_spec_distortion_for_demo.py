import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import os
import sys


#加载测试label数据
mat_src_path = './data'
#print(os.listdir(mat_src_path))
mat_src_path = './data/test_lowlight/origin/soup_bigcorn_orange_1ms.mat'
test_label_hsi = scio.loadmat(mat_src_path)['label']
print(test_label_hsi.shape)

test_lowlight_hsi = scio.loadmat(mat_src_path)['lowlight']
print(test_lowlight_hsi.shape)
height, width, band_num = test_label_hsi.shape

x = np.arange(0, band_num, 1)  

position = (20, 20)

spectral_distortion_label = test_label_hsi[position[0], position[1],:]

plt.plot(x, spectral_distortion_label, label='Label')

# 设置题目与坐标轴名称  
plt.ylabel('Reflectance')  
plt.xlabel('Spectral bands') 
 
# 设置图例（置于右下方）  
plt.legend(loc='upper left', fontsize=5) #fontsize可以调整图例的大小 

plt.ylim(0,1.2)
plt.show()

