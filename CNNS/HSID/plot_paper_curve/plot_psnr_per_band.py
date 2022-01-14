import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import os
import sys
from metrics import PSNR, SSIM, SAM

#x = np.linspace(0, 12, 121)  
#plt.plot(x, np.sin(x))
#plt.show()

print(os.getcwd())
#加载测试label数据
mat_src_path = './data'
#print(os.listdir(mat_src_path))
mat_src_path = './data/test_lowlight/origin/soup_bigcorn_orange_1ms.mat'
test_label_hsi = scio.loadmat(mat_src_path)['label']
print(test_label_hsi.shape)

#加载测试rdn方法的测试结果数据
rdn_eca_src_path = './data/testresult/indoor_mat/lhsie_indoorlhsie_indoor_result.mat'
rdn_eca_hsi = scio.loadmat(rdn_eca_src_path)['denoised']
print(rdn_eca_hsi.shape)

#加载测试encam方法的测试结果数据
encam_src_path = './data/testresult/result_encam.mat'
encam_hsi = scio.loadmat(encam_src_path)['denoised']

#加载测试encam方法的测试结果数据
hsid_src_path = './data/testresult/indoor_mat/hsid_origin3d_try2_result.mat'
hsid_hsi = scio.loadmat(hsid_src_path)['denoised']

#加载测试LRTV方法的测试结果数据
LRTV_src_path = './data/testresult/LRTV_enhanced.mat'
LRTV_hsi = scio.loadmat(LRTV_src_path)['denoised']

#加载测试BM4D方法的测试结果数据
BM4D_src_path = './data/testresult/BM4D_enhanced.mat'
BM4D_hsi = scio.loadmat(BM4D_src_path)['denoised']

#加载测试LRMR方法的测试结果数据
LRMR_src_path = './data/testresult/LRMR_enhanced.mat'
LRMR_hsi = scio.loadmat(LRMR_src_path)['denoised']

#加载测试LRTA方法的测试结果数据
LRTA_src_path = './data/testresult/LRTA_enhanced.mat'
LRTA_hsi = scio.loadmat(LRTA_src_path)['denoised']

#加载测试histeq方法的测试结果数据
HISTEQ_src_path = './data/testresult/HISTEQ_enhanced.mat'
HISTEQ_hsi = scio.loadmat(HISTEQ_src_path)['denoised']

#
#加载测试MSR方法的测试结果数据
MSR_src_path = './data/testresult/indoor_mat/MSR_enhanced.mat'
MSR_hsi = scio.loadmat(MSR_src_path)['denoised']

#CLAHE_enhanced
CLAHE_enhanced_path = './data/testresult/indoor_mat/CLAHE_enhanced.mat'
CLAHE_hsi = scio.loadmat(CLAHE_enhanced_path)['denoised']

#RetinexNet_enhanced
RetinexNet = './data/testresult/indoor_mat/RetinexNet_indoor_lowlight_enhanced.mat'
RetinexNet_hsi = scio.loadmat(RetinexNet)['denoised']

#RUAS_enhanced
RUASNet = './data/testresult/indoor_mat/RUAS_enhanced.mat'
RUASNet_hsi = scio.loadmat(RUASNet)['denoised']

height, width, band_num = rdn_eca_hsi.shape
print(band_num)

psnrlist_rdneca = []
psnrlist_encam = []
psnrlist_hsid = []
psnrlist_LRTV = []
psnrlist_BM4D = []
psnrlist_LRMR = []
psnrlist_LRTA = []
psnrlist_HISTEQ = []
psnrlist_MSR = []
psnrlist_CLAHE = []
psnrlist_RetinexNet = []
psnrlist_RUASNet = []

for i in range(band_num):
    enlightened_band = rdn_eca_hsi[:,:,i]
    label_band = test_label_hsi[:,:,i]
    psnrlist_rdneca.append(PSNR(enlightened_band, label_band))
    psnrlist_encam.append(PSNR(encam_hsi[:,:,i], label_band))
    psnrlist_hsid.append(PSNR(hsid_hsi[:,:,i], label_band))
    psnrlist_LRTV.append(PSNR(LRTV_hsi[:,:,i], label_band))
    psnrlist_BM4D.append(PSNR(BM4D_hsi[:,:,i], label_band))
    psnrlist_LRMR.append(PSNR(LRMR_hsi[:,:,i], label_band))
    psnrlist_LRTA.append(PSNR(LRTA_hsi[:,:,i], label_band))
    psnrlist_HISTEQ.append(PSNR(HISTEQ_hsi[:,:,i], label_band))
    psnrlist_MSR.append(PSNR(MSR_hsi[:,:,i], label_band))
    psnrlist_CLAHE.append(PSNR(CLAHE_hsi[:,:,i], label_band))
    psnrlist_RetinexNet.append(PSNR(RetinexNet_hsi[:,:,i], label_band))
    psnrlist_RUASNet.append(PSNR(RUASNet_hsi[:,:,i], label_band))
x = np.arange(0, band_num, 1)  
print(x)


# 创建图像  

plt.plot(x, psnrlist_rdneca, color='red', label='LHSIE(ours)')
plt.plot(x, psnrlist_encam, color='green', label='ENCAM')
plt.plot(x, psnrlist_hsid, color='blue', label='HSID-CNN')
plt.plot(x, psnrlist_RetinexNet, color='orange', label='Retinex-Net')
plt.plot(x, psnrlist_MSR, color='purple', label='MSR')
plt.plot(x, psnrlist_HISTEQ, color='brown', label='HE')
plt.plot(x, psnrlist_BM4D, color='pink', label='BM4D')
plt.plot(x, psnrlist_CLAHE, color='olive', label='CLAHE')
#plt.plot(x, psnrlist_RUASNet, label='RUASNet')

# 设置题目与坐标轴名称  
plt.ylabel('PSNR')  
plt.xlabel('Band Number') 
 
# 设置图例（置于右下方）  
plt.legend(loc='upper left', fontsize=5) #fontsize可以调整图例的大小 

plt.ylim(1,60)
plt.show()