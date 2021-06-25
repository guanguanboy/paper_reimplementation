import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_relu(inchannels, outchannels):
    layer = [nn.Conv2d(inchannels, outchannels, 3, 1, 1), nn.ReLU(inplace=True)]
    return layer

class HSIDRefactored(nn.Module):
    def __init__(self, k=24):
        super(HSIDRefactored, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu

        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, 60))
        self.conv2_3 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv4_5 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv6_7 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv8_9 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))     

        self.feature_conv3 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv5 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv7 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
        self.feature_conv9 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
         
        #self.feature_conv_3_5_7_9 = concat
        self.conv10 = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x_spatial, x_spectral):
        x_spatial_feature_3 = self.spatial_feature_3(x_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(x_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(x_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(x_spectral)
        x_spectral_feature_5 = self.spectral_feature_5(x_spectral)
        x_spectral_feature_7 = self.spectral_feature_7(x_spectral)

        feature_3_5_7 = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7), dim=1) #在通道维concat
        feature_3_5_7 = self.relu(feature_3_5_7)
        #print('feature_3_5_7 shape =', feature_3_5_7.shape)

        feature_3_5_7_2 = torch.cat((x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1) # 在通道维concat
        feature_3_5_7_2 = self.relu(feature_3_5_7_2)
        #print('feature_3_5_7_2 shape =', feature_3_5_7_2.shape)

        feature_all = torch.cat((feature_3_5_7, feature_3_5_7_2), dim=1)
        #print('feature_all shape =', feature_all.shape)

        x1 = self.conv1(feature_all)
        x3 = self.conv2_3(x1)
        x5 = self.conv4_5(x3)
        x7 = self.conv6_7(x5)
        x9 = self.conv8_9(x7)
        
        feature_conv3 = self.feature_conv3(x3)
        feature_conv5 = self.feature_conv5(x5)
        feature_conv7 = self.feature_conv7(x7)
        feature_conv9 = self.feature_conv9(x9)

        feature_conv_3_5_7_9 = torch.cat((feature_conv3, feature_conv5, feature_conv7, feature_conv9), dim=1)
        
        feature_conv_3_5_7_9 = self.relu(feature_conv_3_5_7_9)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        output = self.conv10(feature_conv_3_5_7_9)

        return output # + x_spatial


def test():
    net = HSIDRefactored(24)
    print(net)

    data = torch.randn(1, 24, 200, 200)
    data1 = torch.randn(1, 1, 200, 200)

    output = net(data1, data)
    print(output.shape)


class HSIDInception(nn.Module):
    def __init__(self, k=24):
        super(HSIDInception, self).__init__()
        self.spatial_feature_3 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)))

        self.spatial_feature_5 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=(1,5), stride=1, padding=(0,2)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(5,1), stride=1, padding=(2,0)))

        self.spatial_feature_7 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=(1,7), stride=1, padding=(0,3)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(7,1), stride=1, padding=(3,0)))

        self.spectral_feature_3 = nn.Sequential(nn.Conv2d(k, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)))
        self.spectral_feature_5 = nn.Sequential(nn.Conv2d(k, 20, kernel_size=(1,5), stride=1, padding=(0,2)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(5,1), stride=1, padding=(2,0)))
        self.spectral_feature_7 = nn.Sequential(nn.Conv2d(k, 20, kernel_size=(1,7), stride=1, padding=(0,3)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(7,1), stride=1, padding=(3,0)))

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu

        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, 60))
        self.conv2_3 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv4_5 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv6_7 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv8_9 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))     

        self.feature_conv3 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv5 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv7 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
        self.feature_conv9 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
         
        #self.feature_conv_3_5_7_9 = concat
        self.conv10 = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x_spatial, x_spectral):
        x_spatial_feature_3 = self.spatial_feature_3(x_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(x_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(x_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(x_spectral)
        x_spectral_feature_5 = self.spectral_feature_5(x_spectral)
        x_spectral_feature_7 = self.spectral_feature_7(x_spectral)

        feature_3_5_7 = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7), dim=1) #在通道维concat
        feature_3_5_7 = self.relu(feature_3_5_7)
        #print('feature_3_5_7 shape =', feature_3_5_7.shape)

        feature_3_5_7_2 = torch.cat((x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1) # 在通道维concat
        feature_3_5_7_2 = self.relu(feature_3_5_7_2)
        #print('feature_3_5_7_2 shape =', feature_3_5_7_2.shape)

        feature_all = torch.cat((feature_3_5_7, feature_3_5_7_2), dim=1)
        #print('feature_all shape =', feature_all.shape)

        x1 = self.conv1(feature_all)
        x3 = self.conv2_3(x1)
        x5 = self.conv4_5(x3)
        x7 = self.conv6_7(x5)
        x9 = self.conv8_9(x7)
        
        feature_conv3 = self.feature_conv3(x3)
        feature_conv5 = self.feature_conv5(x5)
        feature_conv7 = self.feature_conv7(x7)
        feature_conv9 = self.feature_conv9(x9)

        feature_conv_3_5_7_9 = torch.cat((feature_conv3, feature_conv5, feature_conv7, feature_conv9), dim=1)
        
        feature_conv_3_5_7_9 = self.relu(feature_conv_3_5_7_9)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        output = self.conv10(feature_conv_3_5_7_9)

        return output # + x_spatial

class HSIDInceptionV2(nn.Module):
    def __init__(self, k=24):
        super(HSIDInceptionV2, self).__init__()
        self.spatial_feature_3 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)))

        self.spatial_feature_5 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0))
                                                )

        self.spatial_feature_7 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0))
                                                )

        self.spectral_feature_3 = nn.Sequential(nn.Conv2d(k, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)))
        self.spectral_feature_5 = nn.Sequential(nn.Conv2d(k, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0))
                                                )
        self.spectral_feature_7 = nn.Sequential(nn.Conv2d(k, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)))

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu

        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, 60))
        self.conv2_3 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv4_5 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv6_7 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv8_9 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))     

        self.feature_conv3 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv5 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv7 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
        self.feature_conv9 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
         
        #self.feature_conv_3_5_7_9 = concat
        self.conv10 = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x_spatial, x_spectral):
        x_spatial_feature_3 = self.spatial_feature_3(x_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(x_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(x_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(x_spectral)
        x_spectral_feature_5 = self.spectral_feature_5(x_spectral)
        x_spectral_feature_7 = self.spectral_feature_7(x_spectral)

        feature_3_5_7 = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7), dim=1) #在通道维concat
        feature_3_5_7 = self.relu(feature_3_5_7)
        #print('feature_3_5_7 shape =', feature_3_5_7.shape)

        feature_3_5_7_2 = torch.cat((x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1) # 在通道维concat
        feature_3_5_7_2 = self.relu(feature_3_5_7_2)
        #print('feature_3_5_7_2 shape =', feature_3_5_7_2.shape)

        feature_all = torch.cat((feature_3_5_7, feature_3_5_7_2), dim=1)
        #print('feature_all shape =', feature_all.shape)

        x1 = self.conv1(feature_all)
        x3 = self.conv2_3(x1)
        x5 = self.conv4_5(x3)
        x7 = self.conv6_7(x5)
        x9 = self.conv8_9(x7)
        
        feature_conv3 = self.feature_conv3(x3)
        feature_conv5 = self.feature_conv5(x5)
        feature_conv7 = self.feature_conv7(x7)
        feature_conv9 = self.feature_conv9(x9)

        feature_conv_3_5_7_9 = torch.cat((feature_conv3, feature_conv5, feature_conv7, feature_conv9), dim=1)
        
        feature_conv_3_5_7_9 = self.relu(feature_conv_3_5_7_9)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        output = self.conv10(feature_conv_3_5_7_9)

        return output # + x_spatial

class HSIDInceptionV3(nn.Module):
    def __init__(self, k=24):
        super(HSIDInceptionV3, self).__init__()
        self.spatial_feature_1 = nn.Conv2d(1, 20, kernel_size=1, stride=1)

        self.spatial_feature_3 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)))

        self.spatial_feature_5 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0))
                                                )

        self.spatial_feature_7 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0))
                                                )

        self.spectral_feature_1 = nn.Conv2d(k, 40, kernel_size=1, stride=1)

        self.spectral_feature_3 = nn.Sequential(nn.Conv2d(k, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)))
        self.spectral_feature_5 = nn.Sequential(nn.Conv2d(k, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0))
                                                )
        self.spectral_feature_7 = nn.Sequential(nn.Conv2d(k, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(1,3), stride=1, padding=(0,1)),
                                                nn.ReLU(),
                                                nn.Conv2d(20, 20, kernel_size=(3,1), stride=1, padding=(1,0)))

        self.conv0 = nn.Sequential(*conv_relu(180, 120))
        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu

        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, 60))
        self.conv2_3 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv4_5 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv6_7 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv8_9 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))     

        self.feature_conv3 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv5 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv7 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
        self.feature_conv9 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
         
        #self.feature_conv_3_5_7_9 = concat
        self.conv10 = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x_spatial, x_spectral):
        x_spatial_feature_1 = self.spatial_feature_1(x_spatial)
        x_spatial_feature_3 = self.spatial_feature_3(x_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(x_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(x_spatial)

        x_spectral_feature_1 = self.spectral_feature_1(x_spectral)
        x_spectral_feature_3 = self.spectral_feature_3(x_spectral)
        x_spectral_feature_5 = self.spectral_feature_5(x_spectral)
        x_spectral_feature_7 = self.spectral_feature_7(x_spectral)

        feature_3_5_7 = torch.cat((x_spatial_feature_1, x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7), dim=1) #在通道维concat
        feature_3_5_7 = self.relu(feature_3_5_7)
        #print('feature_3_5_7 shape =', feature_3_5_7.shape)

        feature_3_5_7_2 = torch.cat((x_spectral_feature_1, x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1) # 在通道维concat
        feature_3_5_7_2 = self.relu(feature_3_5_7_2)
        #print('feature_3_5_7_2 shape =', feature_3_5_7_2.shape)

        feature_all = torch.cat((feature_3_5_7, feature_3_5_7_2), dim=1)
        #print('feature_all shape =', feature_all.shape)

        x0 = self.conv0(feature_all)
        x1 = self.conv1(x0)
        x3 = self.conv2_3(x1)
        x5 = self.conv4_5(x3)
        x7 = self.conv6_7(x5)
        x9 = self.conv8_9(x7)
        
        feature_conv3 = self.feature_conv3(x3)
        feature_conv5 = self.feature_conv5(x5)
        feature_conv7 = self.feature_conv7(x7)
        feature_conv9 = self.feature_conv9(x9)

        feature_conv_3_5_7_9 = torch.cat((feature_conv3, feature_conv5, feature_conv7, feature_conv9), dim=1)
        
        feature_conv_3_5_7_9 = self.relu(feature_conv_3_5_7_9)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        output = self.conv10(feature_conv_3_5_7_9)

        return output # + x_spatial

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(input_channels, num_channels):#只变换通道数，不变换size
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1))

def create_denseModel():

    b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    # `num_channels`为当前的通道数
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间添加一个转换层，使通道数量减半
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))

    return net

def create_denoisedenseModel():
    # `num_channels`为当前的通道数
    num_channels, growth_rate = 60, 20
    num_convs_in_dense_blocks = [3, 3, 3, 3] #添加了两个dense_block
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间添加一个转换层，使通道数量减半
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

    net = nn.Sequential(*blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),transition_block(num_channels, num_channels // 2))

    return net


def denseblock_test():
    blk = DenseBlock(2, 3, 10)
    X = torch.randn(4, 3, 8, 8)
    Y = blk(X)
    print(Y.shape)

    blk = transition_block(23, 10)
    print(blk(Y).shape)

    net = create_denoisedenseModel()
    input = torch.randn(50, 60, 20, 20)

    output = net(input)
    print(output.shape)


class HSIDDenseNetTwoStage(nn.Module):
    def __init__(self, k=24):
        super(HSIDDenseNetTwoStage, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu

        #self.feature_all : Concat
        self.conv1 = nn.Sequential(*conv_relu(120, 60))
        self.conv2_3 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv4_5 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv6_7 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv8_9 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))     

        self.feature_conv3 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv5 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv7 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
        self.feature_conv9 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
         
        #self.feature_conv_3_5_7_9 = concat
        self.conv10 = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)

        self.densenet = create_denoisedenseModel()

        self.conv10_stage2 = nn.Conv2d(30, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x_spatial, x_spectral):
        x_spatial_feature_3 = self.spatial_feature_3(x_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(x_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(x_spatial)

        x_spectral_feature_3 = self.spectral_feature_3(x_spectral)
        x_spectral_feature_5 = self.spectral_feature_5(x_spectral)
        x_spectral_feature_7 = self.spectral_feature_7(x_spectral)

        feature_3_5_7 = torch.cat((x_spatial_feature_3, x_spatial_feature_5, x_spatial_feature_7), dim=1) #在通道维concat
        feature_3_5_7 = self.relu(feature_3_5_7)
        #print('feature_3_5_7 shape =', feature_3_5_7.shape)

        feature_3_5_7_2 = torch.cat((x_spectral_feature_3, x_spectral_feature_5, x_spectral_feature_7), dim=1) # 在通道维concat
        feature_3_5_7_2 = self.relu(feature_3_5_7_2)
        #print('feature_3_5_7_2 shape =', feature_3_5_7_2.shape)

        feature_all = torch.cat((feature_3_5_7, feature_3_5_7_2), dim=1)
        #print('feature_all shape =', feature_all.shape)

        x1 = self.conv1(feature_all)
        x3 = self.conv2_3(x1)
        x5 = self.conv4_5(x3)
        x7 = self.conv6_7(x5)
        x9 = self.conv8_9(x7)
        
        feature_conv3 = self.feature_conv3(x3)
        feature_conv5 = self.feature_conv5(x5)
        feature_conv7 = self.feature_conv7(x7)
        feature_conv9 = self.feature_conv9(x9)

        feature_conv_3_5_7_9 = torch.cat((feature_conv3, feature_conv5, feature_conv7, feature_conv9), dim=1)
        
        feature_conv_3_5_7_9 = self.relu(feature_conv_3_5_7_9)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        output = self.conv10(feature_conv_3_5_7_9)

        refined_features = self.densenet(feature_conv_3_5_7_9)
        refined_residual = self.conv10_stage2(refined_features)

        return output, refined_residual # + x_spatial

if __name__ == '__main__':
    test()
    denseblock_test()
