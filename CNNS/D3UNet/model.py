import torch
from torch import nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ContractingBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, use_maxpooling=True):
        super(ContractingBlock, self).__init__()

        self.conv3d_1 = nn.Conv3d(in_channels=input_channels, out_channels=hidden_channels, \
            kernel_size=(3, 3, 3),stride=(1, 1, 1), padding=(1, 1, 1))
        self.activation = nn.ReLU()
        self.conv3d_2 = nn.Conv3d(in_channels=hidden_channels, out_channels=output_channels, \
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.max_pooling3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.use_maxpool = use_maxpooling

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.activation(x)
        x = self.conv3d_2(x)
        x = self.activation(x)
        #if self.use_maxpool:
            #x = self.max_pooling3d(x)

        return x


class ExpandingBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, contain_2Conv=True, isFinalLayer=False):
        super(ExpandingBlock, self).__init__()

        self.conv3d_1 = nn.Conv3d(in_channels=input_channels, out_channels=hidden_channels, \
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(in_channels=hidden_channels, out_channels=output_channels, \
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.activation = nn.ReLU()
        self.contain_2conv = contain_2Conv
        self.isFinalLayer = isFinalLayer
        self.activation_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv3d_1(x)

        if self.isFinalLayer:
            x = self.activation_sigmoid(x)
        else:
            x = self.activation(x)

        if self.contain_2conv:
            x = self.conv3d_2(x)
            x = self.activation(x)

        return x



class D3UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(D3UNet, self).__init__()

        self.contract1 = ContractingBlock(input_channels, 32, 64, use_maxpooling=True)
        self.contract2 = ContractingBlock(64, 64, 128, use_maxpooling=False)
        self.expand1 = ExpandingBlock(192, 64, 64, contain_2Conv=True)
        self.expand2 = ExpandingBlock(64, output_channels, output_channels, contain_2Conv=False, isFinalLayer=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear') #该函数可以处理3D tensor
        self.max_pooling3d = nn.MaxPool3d(kernel_size=(2, 2, 2))

    def forward(self, x):
        x0 = self.contract1(x)
        print('x0.shape:', x0.shape)

        x1 = self.max_pooling3d(x0)
        print('x1.shape:', x1.shape)

        x2 = self.contract2(x1)
        print('x2.shape:', x2.shape)
        
        x3 = self.upsample(x2)
        print('x3.shape:', x3.shape)
        #这里需要concatnate
        x4 = torch.cat([x3, x0], axis=1)
        print('x4.shape:', x4.shape)
        
        x5 = self.expand1(x4)
        print('x5.shape:', x5.shape)

        x6 = self.expand2(x5)
        print('x6.shape:', x6.shape)

        return x6

net_input = torch.randn(4, 4, 16, 160, 160) #16为depth
#(batch_size, num_channels, depth， height, width)
net_input = net_input.to(DEVICE)
model = D3UNet(4, 3).to(DEVICE)

x = model(net_input)
print(x.shape)
print(model)