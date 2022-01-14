# -*- coding:utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文

rdecab_num = [1, 2, 3, 4, 5, 6]
psnr_list = [38.1896, 38.2811, 38.2303, 38.7380, 38.3344, 38.2442]

#plt.plot(rdecab_num, psnr_list, marker='o', markerfacecolor='white', color='r')

"""
# 另外再说两个参数
markeredgecolor # 圆边缘的颜色
markeredgewidth # 圆的线宽
"""

plt.plot(rdecab_num, psnr_list, color='r')
#for i in range(len(rdecab_num)):

plt.scatter(rdecab_num, psnr_list, color= 'w', marker='o', edgecolors='r', s=40)

# 设置题目与坐标轴名称  
plt.ylabel('PSNR (dB)')  
plt.xlabel('EAB Number') 
plt.gcf().set_facecolor(np.ones(3)* 240 / 255)   # 生成画布的大小
plt.grid()  # 生成网格
plt.show()