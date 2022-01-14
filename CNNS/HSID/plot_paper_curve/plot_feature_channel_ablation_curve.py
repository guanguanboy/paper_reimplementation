# -*- coding:utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文

feat_channel_num = [30, 60,90, 120, 150, 180]

psnr_list = [38.2838,  38.7104,38.4784, 38.7036, 38.7681, 38.7021]

#plt.plot(rdecab_num, psnr_list, marker='o', markerfacecolor='white', color='r')

"""
# 另外再说两个参数
markeredgecolor # 圆边缘的颜色
markeredgewidth # 圆的线宽
"""

plt.plot(feat_channel_num, psnr_list, color='r')
#for i in range(len(rdecab_num)):

plt.scatter(feat_channel_num, psnr_list, color= 'w', marker='o', edgecolors='r', s=40)

# 设置题目与坐标轴名称  
plt.ylabel('PSNR (dB)')  
plt.xlabel('Feature Dimensions in EAB') 
plt.gcf().set_facecolor(np.ones(3)* 240 / 255)   # 生成画布的大小
plt.xticks(feat_channel_num)
plt.grid()  # 生成网格
plt.show()