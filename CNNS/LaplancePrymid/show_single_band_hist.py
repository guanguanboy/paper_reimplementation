from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#src=Image.open('007_2_2021-01-19_044_label.png')
#src=Image.open('label_high_freq.png')
#src=Image.open('label_low_freq.png')
#src=Image.open('lowlight_high_freq.png')
#src=Image.open('lowlight_low_freq.png')
#src = Image.open('lowlight_high_freq_r_chl.png')
src = Image.open('lowlight_high_freq_all_chl_sum.png')

plt.figure("lena")
ar=np.array(src).flatten()
plt.hist(ar, bins=256,facecolor='black',edgecolor='black',stacked=1)
plt.show()