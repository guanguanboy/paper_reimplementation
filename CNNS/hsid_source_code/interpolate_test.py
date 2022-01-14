import torch

import torch.nn.functional as F

noisy = torch.randn(10, 36, 64, 64)
noisy_half_scale = F.interpolate(noisy, scale_factor=0.5, mode="bilinear")
print(noisy_half_scale.shape)
