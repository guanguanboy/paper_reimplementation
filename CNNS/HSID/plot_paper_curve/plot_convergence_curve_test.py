import numpy as np
import os
import matplotlib.pyplot as plt

print(os.getcwd())

mpsnr_per_channel_l1loss_list = np.load('./plot_paper_curve/mpsnr_per_epoch_outdoor.npy')
mpsnr_per_channel_hsid_origin = np.load('./plot_paper_curve/mpsnr_per_epoch_hsid_origin_outdoor.npy')

list_size = len(mpsnr_per_channel_l1loss_list)
list_size_hsid_origin = len(mpsnr_per_channel_hsid_origin)

epoch_count = np.arange(0, list_size, 1) + 1  
plt.plot(epoch_count, mpsnr_per_channel_l1loss_list, label='L1 Loss')
print(mpsnr_per_channel_l1loss_list)
epoch_count_hsid_origin = np.arange(0, list_size_hsid_origin, 1) + 1  
plt.plot(epoch_count_hsid_origin, mpsnr_per_channel_hsid_origin, label='hsid_origin_L2 Loss')

# 设置题目与坐标轴名称  
plt.ylabel('MPSNR')  
plt.xlabel('Epoch Number') 
 
# 设置图例（置于右下方）  
plt.legend(loc='lower right', fontsize=8) #fontsize可以调整图例的大小 

#plt.ylim(1,60)
plt.show()
