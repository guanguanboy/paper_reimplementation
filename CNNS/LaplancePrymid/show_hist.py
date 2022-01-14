#https://www.cnblogs.com/denny402/p/5096790.html
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

"""
img=np.array(Image.open('dog1.jpg').convert('L'))

plt.figure("lena")
arr=img.flatten()
n, bins, patches = plt.hist(arr, bins=256, normed=1, facecolor='green', alpha=0.75)  
plt.show()
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
src=Image.open('007_2_2021-01-19_044_label.png')
#src=Image.open('007_2_2021-01-19_044_lowlight.png')
#src=Image.open('label_high_freq.png')
#src=Image.open('label_low_freq.png')
#src=Image.open('lowlight_high_freq.png')
#src=Image.open('lowlight_low_freq.png')
#src = Image.open('lowlight_high_freq_r_chl.png')
r,g,b=src.split()

print(np.array(r))
plt.figure("lena")
ar=np.array(r).flatten()
plt.hist(ar, bins=256,facecolor='r',edgecolor='r',stacked=1)
ag=np.array(g).flatten()
plt.hist(ag, bins=256, facecolor='g',edgecolor='g',stacked=1)
ab=np.array(b).flatten()
plt.hist(ab, bins=256, facecolor='b',edgecolor='b')
plt.show()