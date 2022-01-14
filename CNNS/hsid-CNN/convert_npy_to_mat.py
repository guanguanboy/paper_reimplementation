import numpy as np
import scipy.io as scio

#test datas
test=np.load('test_washington.npy') #里面的内容与GT_crop.mat是完全一样的。

print(test.shape)

scio.savemat('test_washington.mat', {'saved_temp': test})

