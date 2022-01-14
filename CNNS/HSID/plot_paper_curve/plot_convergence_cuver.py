import numpy as np
import os
import matplotlib.pyplot as plt

print(os.getcwd())

mpsnr_per_channel_l1loss_list = np.load('./plot_paper_curve/mpsnr_per_epoch_l1loss.npy')

mpsnr_per_channel_list = np.load('./plot_paper_curve/mpsnr_per_epoch.npy')

print(mpsnr_per_channel_list.shape)
list_size = len(mpsnr_per_channel_list)

epoch_count = np.arange(0, list_size, 1) + 1  
print(epoch_count)

plt.plot(epoch_count, mpsnr_per_channel_list, label='L2 Loss')
plt.plot(epoch_count, mpsnr_per_channel_l1loss_list, label='L1 Loss', linestyle='dashed')

# 设置题目与坐标轴名称  
plt.ylabel('MPSNR')  
plt.xlabel('Epoch Number') 
 
# 设置图例（置于右下方）  
plt.legend(loc='lower right', fontsize=8) #fontsize可以调整图例的大小 

#plt.ylim(1,60)
plt.show()
