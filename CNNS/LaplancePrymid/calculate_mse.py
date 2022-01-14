import math
import numpy as np
import cv2

def cal_mse(pred, gt):
      
    valid = gt - pred
    mse = np.mean(valid ** 2)

    return mse


o = cv2.imread("007_2_2021-01-19_044_label.png", cv2.IMREAD_UNCHANGED)
o_lowlight = cv2.imread("007_2_2021-01-19_044_lowlight.png", cv2.IMREAD_UNCHANGED)

mse_result = cal_mse(o_lowlight, o)
print(mse_result)

o = cv2.imread("label_high_freq.png", cv2.IMREAD_UNCHANGED)
o_lowlight = cv2.imread("lowlight_high_freq.png", cv2.IMREAD_UNCHANGED)

mse_result = cal_mse(o_lowlight, o)
print(mse_result)

o = cv2.imread("label_low_freq.png", cv2.IMREAD_UNCHANGED)
o_lowlight = cv2.imread("lowlight_low_freq.png", cv2.IMREAD_UNCHANGED)

mse_result = cal_mse(o_lowlight, o)
print(mse_result)

o = cv2.imread("hist_label_high_freq_b_chl.png", cv2.IMREAD_UNCHANGED)
o_lowlight = cv2.imread("hist_lowlight_high_freq_all_chl_sum.png", cv2.IMREAD_UNCHANGED)

mse_result = cal_mse(o_lowlight, o)
print(mse_result)