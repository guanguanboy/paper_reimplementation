import torch
import torch.nn as nn

in_channel = 3
out_channel = 9

stride = 1
groups = 3
padding = 0
kernel_size = 3

input_matrix = torch.randn(10, 3, 6, 6)#batchsize , channel_num, width, height
print(f'intput matrix shape: {input_matrix.shape}')



normal_conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
output_normal = normal_conv(input_matrix)
print('normal convolution output shape: ', output_normal.shape) #torch.Size([10, 3, 4, 4])

#普通的卷积，如果groups==out_channel==in_channel
depthwise_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=0, groups=in_channel)
output = depthwise_conv(input_matrix)

print('depthwise_conv output shape: ', output.shape)


"""
class depthwise_conv2d(nn.Module):
    def __init__(self, n_in, n_out):
        super(depthwise_conv2d, self).__init__()
        self.depth_wise = nn.Conv2d(n_in, n_in, kernel_size=3, padding=1, groups=n_in)
        self.point_wise = nn.Conv2d(n_in, n_out, kernel_size=1) #其实就是卷积核为1的普通卷积

    def forward(self, x):
        out = self.depth_wise(x)
        print('out shape:', out.shape)
        out = self.point_wise(out)
        return out

input_matrix = torch.randn(10, 3, 6, 6)
depth_wise_conv_obj = depthwise_conv2d(3, 3)
print(depth_wise_conv_obj)
output = depth_wise_conv_obj(input_matrix)
print(output.shape)
"""