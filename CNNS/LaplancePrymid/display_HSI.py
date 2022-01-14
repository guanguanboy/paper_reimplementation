from spectral import *
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


mat_data = scipy.io.loadmat('007_2_2021-01-19_044_lowlight.mat')
print(mat_data['lowlight_normalized_hsi'].shape)

lowlight_cubic = mat_data['lowlight_normalized_hsi']
#img = open_image('../../../../DataSets/hyperspectraldatasets/outdown_data/origin/1ms/007_2_2021-01-19_044/capture/DARKREF_007_2_2021-01-19_044.hdr')

img = lowlight_cubic

view_cube(img, bands=[29, 19, 9])

thisIsLove = input()
if thisIsLove:
    print("True")
