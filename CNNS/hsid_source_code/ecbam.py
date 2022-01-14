import torch
import math
import torch.nn as nn
from torch.nn import init

class ChannelAttention(nn.Module):
    def __init__(self, k_size=3):
        super(ChannelAttention, self).__init__()

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg_y = self.avg_pool(x) ##bs,c,1,1
        max_y = self.max_pool(x) #bs,c,1,1

        y = avg_y + max_y
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class ECBAMBlock(nn.Module):

    def __init__(self, chan_k_size=3, spatial_k_size=3):
        super().__init__()
        self.ca=ChannelAttention(k_size=chan_k_size)
        self.sa=SpatialAttention(kernel_size=spatial_k_size)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual



if __name__ == '__main__':

    img = torch.zeros(128, 64, 20, 20)
    net = ECBAMBlock()
    out = net(img)
    print(out.size()) #torch.Size([128, 64, 20, 20])