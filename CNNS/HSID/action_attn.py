import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb


class Action(nn.Module):
    def __init__(self, net, n_segment=3, shift_div=8):
        super(Action, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.in_channels = self.net.in_channels
        self.out_channels = self.net.out_channels
        self.kernel_size = self.net.kernel_size
        self.stride = self.net.stride
        self.padding = self.net.padding
        self.reduced_channels = self.in_channels//16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fold = self.in_channels // shift_div

        # shifting
        self.action_shift = nn.Conv1d(
                                    self.in_channels, self.in_channels,
                                    kernel_size=3, padding=1, groups=self.in_channels,
                                    bias=False)      
        self.action_shift.weight.requires_grad = True
        self.action_shift.weight.data.zero_()
        self.action_shift.weight.data[:self.fold, 0, 2] = 1 # shift left
        self.action_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right  


        if 2*self.fold < self.in_channels:
            self.action_shift.weight.data[2 * self.fold:, 0, 1] = 1 # fixed


        # # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), 
                                    stride=(1, 1 ,1), bias=False, padding=(1, 1, 1))       


        # # channel excitation
        self.action_p2_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, 
                                       groups=1)
        self.action_p2_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
    


        # motion excitation
        self.pad = (0,0,0,0,0,0,0,1)
        self.action_p3_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p3_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.action_p3_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 3), 
                                    stride=(1 ,1), bias=False, padding=(1, 1), groups=self.reduced_channels)
        self.action_p3_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        print('=> Using ACTION')


    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        x_shift = x.view(n_batch, self.n_segment, c, h, w)
        x_shift = x_shift.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x_shift = x_shift.contiguous().view(n_batch*h*w, c, self.n_segment)
        x_shift = self.action_shift(x_shift)  # (n_batch*h*w, c, n_segment)
        x_shift = x_shift.view(n_batch, h, w, c, self.n_segment)
        x_shift = x_shift.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x_shift = x_shift.contiguous().view(nt, c, h, w)


        # 3D convolution: c*T*h*w, spatial temporal excitation
        nt, c, h, w = x_shift.size()
        x_p1 = x_shift.view(n_batch, self.n_segment, c, h, w).transpose(2,1).contiguous()
        x_p1 = x_p1.mean(1, keepdim=True)
        x_p1 = self.action_p1_conv1(x_p1)
        x_p1 = x_p1.transpose(2,1).contiguous().view(nt, 1, h, w)
        x_p1 = self.sigmoid(x_p1)
        x_p1 = x_shift * x_p1 + x_shift


        # 2D convolution: c*T*1*1, channel excitation
        x_p2 = self.avg_pool(x_shift)
        x_p2 = self.action_p2_squeeze(x_p2)
        nt, c, h, w = x_p2.size()
        x_p2 = x_p2.view(n_batch, self.n_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()
        x_p2 = self.action_p2_conv1(x_p2)
        x_p2 = self.relu(x_p2)
        x_p2 = x_p2.transpose(2,1).contiguous().view(-1, c, 1, 1)
        x_p2 = self.action_p2_expand(x_p2)
        x_p2 = self.sigmoid(x_p2)
        x_p2 = x_shift * x_p2 + x_shift
        

        # # 2D convolution: motion excitation
        x3 = self.action_p3_squeeze(x_shift)
        x3 = self.action_p3_bn1(x3)
        nt, c, h, w = x3.size()
        x3_plus0, _ = x3.view(n_batch, self.n_segment, c, h, w).split([self.n_segment-1, 1], dim=1)
        x3_plus1 = self.action_p3_conv1(x3)

        _ , x3_plus1 = x3_plus1.view(n_batch, self.n_segment, c, h, w).split([1, self.n_segment-1], dim=1)
        x_p3 = x3_plus1 - x3_plus0
        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)
        x_p3 = self.avg_pool(x_p3.view(nt, c, h, w))
        x_p3 = self.action_p3_expand(x_p3)
        x_p3 = self.sigmoid(x_p3)
        x_p3 = x_shift * x_p3 + x_shift

        out = self.net(x_p1 + x_p2 + x_p3)
        return out

class SSAttn(nn.Module):
    def __init__(self):
        super(SSAttn, self).__init__()
        # # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), 
                                    stride=(1, 1 ,1), bias=False, padding=(1, 1, 1))   
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x_shift):
        # 3D convolution: c*T*h*w, spatial temporal excitation
        nt, sp, c, h, w = x_shift.size() #torch.Size([2, 36, 20, 32, 32])

        x_p1 = x_shift.transpose(2,1).contiguous() #[2, 20, 36, 32, 32]
        x_p1 = x_p1.mean(1, keepdim=True) #torch.Size([2, 1, 36, 32, 32])
        x_p1 = self.action_p1_conv1(x_p1) ##torch.Size([2, 1, 36, 32, 32])
        x_p1 = x_p1.transpose(2,1).contiguous() #[2, 36, 1, 32, 32]
        x_p1 = self.sigmoid(x_p1)
        #print('atten weight shape =', x_p1.shape)
        x_p1 = x_shift * x_p1 + x_shift    

        return x_p1


class SSAttnSimple(nn.Module):
    def __init__(self):
        super(SSAttnSimple, self).__init__()
        # # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), 
                                    stride=(1, 1 ,1), bias=False, padding=(1, 1, 1))   
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x_shift):
        # 3D convolution: c*T*h*w, spatial temporal excitation
        nt, sp, c, h, w = x_shift.size() #torch.Size([2, 36, 1, 32, 32])

        x_p1 = x_shift.transpose(2,1).contiguous() #[2, 1, 36, 32, 32]
        x_p1 = self.action_p1_conv1(x_p1) ##torch.Size([2, 1, 36, 32, 32])
        x_p1 = x_p1.transpose(2,1).contiguous() #[2, 36, 1, 32, 32]
        x_p1 = self.sigmoid(x_p1)
        #print('atten weight shape =', x_p1.shape)
        x_p1 = x_shift * x_p1 + x_shift    

        return x_p1


def spatial_spectal_attention_test():
    input_t = torch.randn((2, 36, 20, 32, 32))

    ss_attn = SSAttn()

    output = ss_attn(input_t)

    print('ssAttn output shape =', output.shape)

    input_t_sim = torch.randn((2, 36, 1, 32, 32))
    ss_attnsimple = SSAttnSimple()

    output_sim = ss_attnsimple(input_t_sim)
    print('ssattnsimple output shape =', output_sim.shape)



if __name__ == '__main__':
    spatial_spectal_attention_test()