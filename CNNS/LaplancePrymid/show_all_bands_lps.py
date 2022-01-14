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

lowlight_rgb = np.zeros((512, 512, 3))
#lowlight_rgb = [lowlight_cubic[:,:,10], lowlight_cubic[:,:,23], lowlight_cubic[:,:,37]]

lowlight_rgb[:,:,0] = np.floor(lowlight_cubic[:,:,10]*255)
lowlight_rgb[:,:,1] = np.floor(lowlight_cubic[:,:,23]*255)
lowlight_rgb[:,:,2] = np.floor(lowlight_cubic[:,:,37]*255)
lowlight_r = np.floor(lowlight_cubic[:,:,37]*255)

print(lowlight_rgb)

lowlight_laplasi_cubic = np.zeros((512, 512, 64))

for band_idx in range(64):
    band = np.floor(lowlight_cubic[:,:,band_idx]*255)
    od = cv2.pyrDown(band)
    odu = cv2.pyrUp(od)
    lapPyr = band - odu
    lowlight_laplasi_cubic[:,:,band_idx] = lapPyr

print(lapPyr.shape)
print(type(lapPyr))
print(lapPyr)
lowlight_laplasi_cubic=np.rot90(lowlight_laplasi_cubic,k=1, axes=(1,0))

scipy.io.savemat('lowlight_laplance_cubic.mat', {'lowlight_normalized_hsi':lowlight_laplasi_cubic})
lapPry_mean = np.mean(lowlight_laplasi_cubic, axis=2)
print(lapPry_mean.shape)
print(type(lapPry_mean))
#print(lapPry_mean)
hist_low = cv2.calcHist([(lapPry_mean*255).astype(np.uint8)], [0], None, [256], [0, 255])
plt.figure(1)
plt.plot(hist_low, color="r")
plt.title('Low frequence')
plt.show()

cv2.imshow("lapPyr", lapPry_mean)
cv2.waitKey()
cv2.destroyAllWindows()

