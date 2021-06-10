from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        #print(b, c)
        #squeeze produces a channel descriptor by aggregating feature maps across their spatial dimensions
        y = self.avg_pool(x).view(b, c) 
        y = self.fc(y).view(b, c, 1, 1)
        #print('attention map shape:', y.shape) #torch.Size([16, 64, 1, 1])
        return x * y.expand_as(x) # *号表示哈达玛积，既element-wise乘积, 用输入x乘以attention map
