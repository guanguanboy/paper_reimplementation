import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import os
import sys



mat_data = scio.loadmat('007_2_2021-01-19_044_lowlight.mat')
print(mat_data['lowlight_normalized_hsi'].shape)

test_lowlight_hsi = mat_data['lowlight_normalized_hsi']
#o = cv2.imread("007_2_2021-01-19_044_lowlight.png")

mat_label = scio.loadmat('007_2_2021-01-19_044_label.mat')
print(mat_label['label_normalized_hsi'].shape)
test_label_hsi = mat_label['label_normalized_hsi']


test_lowlight_hsi_unit8 =  np.floor(test_lowlight_hsi*255)
test_label_hsi_unit8 =  np.floor(test_label_hsi*255)

height, width, band_num = test_label_hsi.shape

x = np.arange(0, band_num, 1)  

position = (200, 200)

spectral_distortion_label = test_label_hsi_unit8[position[0], position[1],:]
spectral_distortion_label_lowlight = test_lowlight_hsi_unit8[position[0], position[1],:]

plt.plot(x, spectral_distortion_label, label='Label')
plt.plot(x, spectral_distortion_label_lowlight, label='Lowlight')

# 设置题目与坐标轴名称  
plt.ylabel('Reflectance')  
plt.xlabel('Spectral bands') 
 
# 设置图例（置于右下方）  
plt.legend(loc='upper left', fontsize=12) #fontsize可以调整图例的大小 

plt.ylim(0,255)
plt.show()

