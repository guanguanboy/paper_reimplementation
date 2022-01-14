import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


mat_data = scipy.io.loadmat('007_2_2021-01-19_044_lowlight.mat')
print(mat_data['lowlight_normalized_hsi'].shape)

lowlight_cubic = mat_data['lowlight_normalized_hsi']
#o = cv2.imread("007_2_2021-01-19_044_lowlight.png")

mat_label = scipy.io.loadmat('007_2_2021-01-19_044_label.mat')
print(mat_label['label_normalized_hsi'].shape)
#lowlight_cubic = mat_label['label_normalized_hsi']

lowlight_single_band_37 = lowlight_cubic[:,:,37]
lowlight_single_band_37_rot = np.rot90(lowlight_single_band_37,k=1, axes=(1,0))
imsave_path = 'lowlight_single_band_37.png'
plt.imsave(imsave_path, lowlight_single_band_37_rot,cmap="gray")

label_single_band_37 = mat_label['label_normalized_hsi'][:,:,37]
label_single_band_37_rot = np.rot90(label_single_band_37,k=1, axes=(1,0))
imsave_path = 'label_single_band_37.png'
plt.imsave(imsave_path, label_single_band_37_rot, cmap="gray")


