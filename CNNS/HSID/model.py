import torch
import torch.nn as nn
from NLblock import NONLocalBlock2D


class HSID(nn.Module):
    def __init__(self, k=24):
        super(HSID, self).__init__()
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
        self.conv1 = nn.Conv2d(120, 60, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv5 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv9 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        

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
        x1_active = self.relu(x1)
        x2 = self.conv2(x1_active)
        x2_active = self.relu(x2)
        x3 = self.conv3(x2_active)
        x3_active = self.relu(x3)

        x4 = self.conv4(x3_active)
        x4_active = self.relu(x4)
        x5 = self.conv5(x4_active)
        x5_active = self.relu(x5)
        x6 = self.conv6(x5_active)
        x6_active = self.relu(x6)

        x7 = self.conv7(x6_active)
        x7_active = self.relu(x7)
        x8 = self.conv8(x7_active)
        x8_active = self.relu(x8)
        x9 = self.conv9(x8_active)
        x9_active = self.relu(x9)

        feature_conv3 = self.feature_conv3(x3_active)
        feature_conv5 = self.feature_conv5(x5_active)
        feature_conv7 = self.feature_conv7(x7_active)
        feature_conv9 = self.feature_conv9(x9_active)

        feature_conv_3_5_7_9 = torch.cat((feature_conv3, feature_conv5, feature_conv7, feature_conv9), dim=1)
        
        feature_conv_3_5_7_9 = self.relu(feature_conv_3_5_7_9)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        output = self.conv10(feature_conv_3_5_7_9)

        return output # + x_spatial

class HSIDNoLocal(nn.Module):
    def __init__(self, k=24):
        super(HSIDNoLocal, self).__init__()
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
        self.conv1 = nn.Conv2d(120, 60, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv5 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv9 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        

        self.feature_conv3 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv5 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv7 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
        self.feature_conv9 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
         
        #self.feature_conv_3_5_7_9 = concat
        self.conv10 = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)
        self.encab_spatial = NONLocalBlock2D(20, sub_sample=False, bn_layer=False)
        self.encab_feature_conv = NONLocalBlock2D(15, sub_sample=False, bn_layer=False)

    def forward(self, x_spatial, x_spectral):
        x_spatial_feature_3 = self.spatial_feature_3(x_spatial)
        x_spatial_feature_5 = self.spatial_feature_5(x_spatial)
        x_spatial_feature_7 = self.spatial_feature_7(x_spatial)

        #nonLocal操作
        x_spatial_feature_3 = self.encab_spatial(x_spatial_feature_3)
        x_spatial_feature_5 = self.encab_spatial(x_spatial_feature_5)
        x_spatial_feature_7 = self.encab_spatial(x_spatial_feature_7)


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
        x1_active = self.relu(x1)
        x2 = self.conv2(x1_active)
        x2_active = self.relu(x2)
        x3 = self.conv3(x2_active)
        x3_active = self.relu(x3)

        x4 = self.conv4(x3_active)
        x4_active = self.relu(x4)
        x5 = self.conv5(x4_active)
        x5_active = self.relu(x5)
        x6 = self.conv6(x5_active)
        x6_active = self.relu(x6)

        x7 = self.conv7(x6_active)
        x7_active = self.relu(x7)
        x8 = self.conv8(x7_active)
        x8_active = self.relu(x8)
        x9 = self.conv9(x8_active)
        x9_active = self.relu(x9)

        feature_conv3 = self.feature_conv3(x3_active)
        feature_conv5 = self.feature_conv5(x5_active)
        feature_conv7 = self.feature_conv7(x7_active)
        feature_conv9 = self.feature_conv9(x9_active)

        #nonlocal操作
        feature_conv3 = self.encab_feature_conv(feature_conv3)
        feature_conv5 = self.encab_feature_conv(feature_conv5)
        feature_conv7 = self.encab_feature_conv(feature_conv7)
        feature_conv9 = self.encab_feature_conv(feature_conv9)


        feature_conv_3_5_7_9 = torch.cat((feature_conv3, feature_conv5, feature_conv7, feature_conv9), dim=1)
        
        feature_conv_3_5_7_9 = self.relu(feature_conv_3_5_7_9)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        output = self.conv10(feature_conv_3_5_7_9)

        return output # + x_spatial

from selayer import SELayer

class HSIDCA(nn.Module):
    def __init__(self, k=24):
        super(HSIDCA, self).__init__()
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
        self.conv1 = nn.Conv2d(120, 60, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv5 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv9 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        

        self.feature_conv3 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv5 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv7 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
        self.feature_conv9 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
         
        #self.feature_conv_3_5_7_9 = concat
        self.conv10 = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)
        self.selayer = SELayer(60)

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
        x1 = self.selayer(x1)
        x1_active = self.relu(x1)


        x2 = self.conv2(x1_active)
        x2 = self.selayer(x2)
        x2_active = self.relu(x2)


        x3 = self.conv3(x2_active)
        x3 = self.selayer(x3)
        x3_active = self.relu(x3)

        x4 = self.conv4(x3_active)
        x4 = self.selayer(x4)
        x4_active = self.relu(x4)


        x5 = self.conv5(x4_active)
        x5 = self.selayer(x5)
        x5_active = self.relu(x5)

        x6 = self.conv6(x5_active)
        x6 = self.selayer(x6)        
        x6_active = self.relu(x6)

        x7 = self.conv7(x6_active)
        x7 = self.selayer(x7)        
        x7_active = self.relu(x7)

        x8 = self.conv8(x7_active)
        x8 = self.selayer(x8)        
        x8_active = self.relu(x8)

        x9 = self.conv9(x8_active)
        x9 = self.selayer(x9)
        x9_active = self.relu(x9)

        feature_conv3 = self.feature_conv3(x3_active)
        feature_conv5 = self.feature_conv5(x5_active)
        feature_conv7 = self.feature_conv7(x7_active)
        feature_conv9 = self.feature_conv9(x9_active)

        feature_conv_3_5_7_9 = torch.cat((feature_conv3, feature_conv5, feature_conv7, feature_conv9), dim=1)
        feature_conv_3_5_7_9_se = self.selayer(feature_conv_3_5_7_9)
        feature_conv_3_5_7_9 = self.relu(feature_conv_3_5_7_9_se)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        output = self.conv10(feature_conv_3_5_7_9)

        return output # + x_spatial


def test():
    net = HSID(24)
    print(net)

    data = torch.randn(1, 24, 200, 200)
    data1 = torch.randn(1, 1, 200, 200)

    output = net(data1, data)
    print(output.shape)

#test()

