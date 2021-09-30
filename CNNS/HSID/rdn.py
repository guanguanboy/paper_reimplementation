import torch
import torch.nn as nn
import torch.nn.init as init
import math

'''
Residual Dense Network for Image Super-Resolution

Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, Yun Fu

arXiv:1802.08797 [cs.CV]

https://arxiv.org/abs/1802.08797
'''

class BasicBlock(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(BasicBlock,self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels = input_dim,out_channels = output_dim,kernel_size=3,padding=1,stride=1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x,out),1)


class RDB(nn.Module):
    def __init__(self,nb_layers,input_dim,growth_rate):
        super(RDB,self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels = input_dim+nb_layers*growth_rate,\
                                    out_channels =growth_rate,\
                                    kernel_size = 1,\
                                    stride=1,\
                                    padding=0  )
    def _make_layer(self,nb_layers,input_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out) 
        return out+x

class RDN(nn.Module):
    def __init__(self,channel,growth_rate,rdb_number,upscale_factor):
        super(RDN,self).__init__()
        self.SFF1 = nn.Conv2d(in_channels = channel,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        self.SFF2 = nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        self.RDB1 = RDB(nb_layers = rdb_number,input_dim=64,growth_rate=64)
        self.RDB2 = RDB(nb_layers = rdb_number,input_dim=64,growth_rate=64)
        self.RDB3 = RDB(nb_layers = rdb_number,input_dim=64,growth_rate=64)
        self.GFF1 = nn.Conv2d(in_channels = 64*3,out_channels = 64,kernel_size = 1,padding = 0 )
        self.GFF2 = nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1 )
        self.upconv = nn.Conv2d(in_channels = 64, out_channels=(64*upscale_factor*upscale_factor),kernel_size = 3,padding = 1)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor)
        self.conv2 = nn.Conv2d(in_channels = 64,out_channels = channel,kernel_size = 3,padding = 1 )

        self.criterion = nn.L1Loss() 
        self.optimizer = torch.optim.Adam(self.parameters())
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)

    def forward(self,x):
        f_ = self.SFF1(x)
        f_0 = self.SFF2(f_)
        f_1 = self.RDB1(f_0)
        f_2 = self.RDB2(f_1)
        f_3 = self.RDB3(f_2)
        f_D = torch.cat((f_1,f_2,f_3),1)
        f_1x1 = self.GFF1(f_D)
        f_GF = self.GFF2(f_1x1)
        f_DF = f_GF + f_
        f_upconv = self.upconv(f_DF)
        f_upscale = self.pixelshuffle(f_upconv)
        f_conv2 = self.conv2(f_upscale)
        return f_conv2


class DenoiseBasicBlock(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DenoiseBasicBlock,self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels = input_dim,out_channels = output_dim,kernel_size=3,padding=1,stride=1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x,out),1)


class DenoiseRDB(nn.Module):
    def __init__(self,nb_layers,input_dim,growth_rate):
        super(DenoiseRDB,self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels = input_dim+nb_layers*growth_rate,\
                                    out_channels =input_dim,\
                                    kernel_size = 1,\
                                    stride=1,\
                                    padding=0  )
    def _make_layer(self,nb_layers,input_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(DenoiseBasicBlock(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out) 
        return out+x


class DenoiseRDN(nn.Module):
    def __init__(self,channel,growth_rate,rdb_number):
        super(DenoiseRDN,self).__init__()
        #self.SFF1 = nn.Conv2d(in_channels = channel,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        #self.SFF2 = nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1 , stride = 1)
        self.RDB1 = DenoiseRDB(nb_layers = rdb_number,input_dim=60,growth_rate=growth_rate)
        self.RDB2 = DenoiseRDB(nb_layers = rdb_number,input_dim=60,growth_rate=growth_rate)
        self.RDB3 = DenoiseRDB(nb_layers = rdb_number,input_dim=60,growth_rate=growth_rate)
        self.RDB4 = DenoiseRDB(nb_layers = rdb_number,input_dim=60,growth_rate=growth_rate)
        self.RDB5 = DenoiseRDB(nb_layers = rdb_number,input_dim=60,growth_rate=growth_rate)
        self.GFF1 = nn.Conv2d(in_channels = channel*5,out_channels = channel,kernel_size = 1,padding = 0 )
        self.GFF2 = nn.Conv2d(in_channels = channel,out_channels = channel,kernel_size = 3,padding = 1 )


    def forward(self,x):

        f_1 = self.RDB1(x)
        f_2 = self.RDB2(f_1)
        f_3 = self.RDB3(f_2)
        f_4 = self.RDB4(f_3)
        f_5 = self.RDB5(f_4)
        f_D = torch.cat((f_1,f_2,f_3, f_4, f_5),1)
        f_1x1 = self.GFF1(f_D)
        f_gf = self.GFF2(f_1x1)
        return f_gf


def rdn_test():
    input_t = torch.randn(5, 60, 20, 20)
    rdn = DenoiseRDN(channel = 60, growth_rate=20, rdb_number = 4)
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in rdn.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in rdn.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    output = rdn(input_t)
    print(output.shape)

def rdb_test():
    input_t = torch.randn(5, 60, 20, 20)
    rdn = DenoiseRDB(nb_layers = 4,input_dim=60,growth_rate=20)
    output = rdn(input_t)
    print(output.shape)

if __name__ == "__main__":
    rdb_test()
    rdn_test()
