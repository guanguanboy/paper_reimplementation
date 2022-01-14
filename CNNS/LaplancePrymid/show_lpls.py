import cv2
o = cv2.imread("007_2_2021-01-19_044_label.png", cv2.IMREAD_UNCHANGED)
#o_lowlight = cv2.imread("007_2_2021-01-19_044_lowlight.png", cv2.IMREAD_UNCHANGED)

o_r = o[:,:,0]
#o_r = o_lowlight[:,:,0]
print(o_r)

od = cv2.pyrDown(o_r)
odu = cv2.pyrUp(od)
lapPyr = o_r - odu

print(lapPyr.shape)
#cv2.imwrite("lowlight_high_freq_r_chl.png", lapPyr) #显示高频信息
cv2.imwrite("lowlight_high_freq_b_chl.png", lapPyr) #显示高频信息
cv2.imshow("lapPyr", lapPyr)
#cv2.imshow("lapPyr", odu) #显示低频信息
cv2.waitKey()
cv2.destroyAllWindows()
