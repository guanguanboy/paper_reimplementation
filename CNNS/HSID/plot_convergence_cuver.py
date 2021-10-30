import numpy as np
import os
save_model_path = './checkpoints/hsirnd_without_multiscale_and_eca'

file_path = os.path.join(save_model_path, "mpsnr_per_epoch.npy")
a=np.load(file_path)
graphTable=a.tolist()
print(graphTable)

