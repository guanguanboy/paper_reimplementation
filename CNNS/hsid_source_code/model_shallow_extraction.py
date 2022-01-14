import torch

import torch.nn as nn


import torch.nn.functional as F
import torch.nn.init as init

from action_attn import SSAttn,SSAttnSimple

class HSIDShallowExtr(nn.Module):
    def __init__(self, k):
        super(HSIDShallowExtr, self).__init__()

        #self.f3_1 = nn.Conv2d(2, 20, 3, 1, 1)
        #self.f5_1 = nn.Conv2d(2, 20, 5, 1, 2)
        #self.f7_1 = nn.Conv2d(2, 20, 7, 1, 3)
        self.f3_1 = nn.Conv2d(1, 20, 3, 1, 1)
        self.f5_1 = nn.Conv2d(1, 20, 5, 1, 2)
        self.f7_1 = nn.Conv2d(1, 20, 7, 1, 3)

        self.f3_2 = nn.Conv3d(1, 20, (3, 3, 3), 1, (1, 1, 1))
        self.f5_2 = nn.Conv3d(1, 20, (5, 5, 5), 1, (2, 2, 2))
        self.f7_2 = nn.Conv3d(1, 20, (7, 7, 7), 1, (3, 3, 3))
    
        self.f3_2_attn = SSAttn()
        self.f5_2_attn = SSAttn()
        self.f7_2_attn = SSAttn()

        self.reduce = nn.Conv3d(k, 1, (3, 3, 3), 1, (1, 1, 1))


    def forward(self, x, y):

        f3_1 = self.f3_1(x)
        f5_1 = self.f5_1(x)
        f7_1 = self.f7_1(x)

        y = y.unsqueeze(1)

        f3_2 = self.f3_2(y)
        f3_2_attn = self.f3_2_attn(f3_2)
        #print(f3_2_attn.shape)

        f5_2 = self.f5_2(y)
        f5_2_attn = self.f5_2_attn(f5_2)
        #print(f5_2_attn.shape)

        f7_2 = self.f7_2(y)
        f7_2_attn = self.f7_2_attn(f7_2)
        #print(f7_2_attn.shape)

        out1 = F.relu(torch.cat((f3_1, f5_1, f7_1), dim=1)) # 12 60 20 20
        out_attn = F.relu(torch.cat((f3_2_attn, f5_2_attn, f7_2_attn), dim=1))
        #print('out_attn.shape', out_attn.shape)

        out_attn = out_attn.transpose(2, 1)
        out_reduce = self.reduce(out_attn)
        #print('out_reduce.shape', out_reduce.shape)
        out2 = out_reduce.squeeze(1) ## 12 60 20 20
        out3 = torch.cat((out1, out2), dim=1) ## 12 120 20 20
        return out3


class HSIDShallowExtrSim(nn.Module):
    def __init__(self, k):
        super(HSIDShallowExtrSim, self).__init__()

        #self.f3_1 = nn.Conv2d(2, 20, 3, 1, 1)
        #self.f5_1 = nn.Conv2d(2, 20, 5, 1, 2)
        #self.f7_1 = nn.Conv2d(2, 20, 7, 1, 3)
        self.f3_1 = nn.Conv2d(1, 20, 3, 1, 1)
        self.f5_1 = nn.Conv2d(1, 20, 5, 1, 2)
        self.f7_1 = nn.Conv2d(1, 20, 7, 1, 3)

        self.f3_2 = nn.Conv3d(1, 20, (k, 3, 3), 1, (0, 1, 1))
        self.f5_2 = nn.Conv3d(1, 20, (k, 5, 5), 1, (0, 2, 2))
        self.f7_2 = nn.Conv3d(1, 20, (k, 7, 7), 1, (0, 3, 3))
    
        self.attnsim = SSAttnSimple()

    def forward(self, x, y):

        f3_1 = self.f3_1(x)
        f5_1 = self.f5_1(x)
        f7_1 = self.f7_1(x)

        y = y.unsqueeze(2)
        #print(y.shape)
        y = self.attnsim(y)
        #print(y.shape)

        y = y.transpose(2, 1)

        f3_2 = self.f3_2(y)
        f5_2 = self.f5_2(y)
        f7_2 = self.f7_2(y)

        out1 = F.relu(torch.cat((f3_1, f5_1, f7_1), dim=1)) # 12 60 20 20
        out2 = F.relu(torch.cat((f3_2, f5_2, f7_2), dim=1)).squeeze(2)

        out3 = torch.cat((out1, out2), dim=1) ## 12 120 20 20
        return out3

def test_shallow_extr():
    net = HSIDShallowExtr(k=36)
    #print(net)
    netsim = HSIDShallowExtrSim(k=36)

    data = torch.randn(1, 36, 200, 200)
    data1 = torch.randn(1, 1, 200, 200)

    out3 = net(data1, data)
    print('out.shape =', out3.shape)

    outsim = netsim(data1, data)
    print('outsim.shape =', outsim.shape)

if __name__ == '__main__':
    test_shallow_extr()
        