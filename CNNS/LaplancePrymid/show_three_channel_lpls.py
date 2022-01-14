import cv2
o = cv2.imread("007_2_2021-01-19_044_label.png", cv2.IMREAD_UNCHANGED)
o_lowlight = cv2.imread("007_2_2021-01-19_044_lowlight.png", cv2.IMREAD_UNCHANGED)
o_r = o[:,:,0]
#o_r = o
#print(o_r)

od = cv2.pyrDown(o)
odu = cv2.pyrUp(od)
lapPyr = o - odu
#print(lapPyr[:,:,0])
print(lapPyr.shape)

od_lowlight = cv2.pyrDown(o_lowlight)
odu_lowlight = cv2.pyrUp(od_lowlight)
lapPyr_low_light = o_lowlight - odu_lowlight
cv2.imshow("lapPyr", lapPyr) #显示高频信息

"""
————————————————
版权声明：本文为CSDN博主「岁月神偷小拳拳」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/u013925378/article/details/94612840

std::vector<int> compression_params;
compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
compression_params.push_back(0);    // 无压缩png.
compression_params.push_back(cv::IMWRITE_PNG_STRATEGY);
compression_params.push_back(cv::IMWRITE_PNG_STRATEGY_DEFAULT);  
"""
cv2.imwrite("label_high_freq.png", lapPyr)
cv2.imwrite("label_low_freq.png", odu)

cv2.imwrite("lowlight_high_freq.png", lapPyr_low_light)
cv2.imwrite("lowlight_low_freq.png", odu_lowlight)

cv2.imshow("lapPyr", lapPyr_low_light) #显示高频信息
#cv2.imshow("lapPyr", odu_lowlight) #显示低频信息
cv2.waitKey()
cv2.destroyAllWindows()
