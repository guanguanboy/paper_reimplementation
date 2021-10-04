# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# https://github.com/sanghyun-son/EDSR-PyTorch

import torch
import torch.nn as nn


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDNForImageDenoising(nn.Module):
    def __init__(self, input_channels, rdnkernelsize):
        super(RDNForImageDenoising, self).__init__()
        G0 = input_channels
        kSize = rdnkernelsize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (8, 6, 32),
            'D': (6, 4, 32)
        }['C']

        # Shallow feature extraction net
        #self.SFENet1 = nn.Conv2d(input_channels, G0, kSize, padding=(kSize-1)//2, stride=1)
        #self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        self.reconstruct = nn.Conv2d(G0, 1, kSize, padding=(kSize-1)//2, stride=1)

    def forward(self, x):
        #f__1 = self.SFENet1(x)
        #x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        #x += f__1

        return x #返回的是残差


def rdn_for_img_denoise_test():

    input_t = torch.randn(5, 60, 20, 20)

    rdn_model = RDNForImageDenoising(input_channels=60, rdnkernelsize=3)

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in rdn_model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in rdn_model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    output = rdn_model(input_t)
    print(output.shape)

if __name__ == "__main__":
    rdn_for_img_denoise_test()