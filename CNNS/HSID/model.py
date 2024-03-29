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
        self.selayer1 = SELayer(60)
        self.selayer2 = SELayer(60)
        self.selayer3 = SELayer(60)
        self.selayer4 = SELayer(60)
        self.selayer5 = SELayer(60)
        self.selayer6 = SELayer(60)
        self.selayer7 = SELayer(60)
        self.selayer8 = SELayer(60)
        self.selayer9 = SELayer(60)
        self.selayer10 = SELayer(60)
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
        x1 = self.selayer1(x1)
        x1_active = self.relu(x1)


        x2 = self.conv2(x1_active)
        x2 = self.selayer2(x2)
        x2_active = self.relu(x2)


        x3 = self.conv3(x2_active)
        x3 = self.selayer3(x3)
        x3_active = self.relu(x3)

        x4 = self.conv4(x3_active)
        x4 = self.selayer4(x4)
        x4_active = self.relu(x4)


        x5 = self.conv5(x4_active)
        x5 = self.selayer5(x5)
        x5_active = self.relu(x5)

        x6 = self.conv6(x5_active)
        x6 = self.selayer6(x6)        
        x6_active = self.relu(x6)

        x7 = self.conv7(x6_active)
        x7 = self.selayer7(x7)        
        x7_active = self.relu(x7)

        x8 = self.conv8(x7_active)
        x8 = self.selayer8(x8)        
        x8_active = self.relu(x8)

        x9 = self.conv9(x8_active)
        x9 = self.selayer9(x9)
        x9_active = self.relu(x9)

        feature_conv3 = self.feature_conv3(x3_active)
        feature_conv5 = self.feature_conv5(x5_active)
        feature_conv7 = self.feature_conv7(x7_active)
        feature_conv9 = self.feature_conv9(x9_active)

        feature_conv_3_5_7_9 = torch.cat((feature_conv3, feature_conv5, feature_conv7, feature_conv9), dim=1)
        feature_conv_3_5_7_9_se = self.selayer10(feature_conv_3_5_7_9)
        feature_conv_3_5_7_9 = self.relu(feature_conv_3_5_7_9_se)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        output = self.conv10(feature_conv_3_5_7_9)

        return output # + x_spatial

from selayer import SEBasicBlock

class HSIDRes(nn.Module):
    def __init__(self, k=24):
        super(HSIDRes, self).__init__()
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
        self.reslayer1 = SEBasicBlock(60,60)
        self.reslayer2 = SEBasicBlock(60, 60)

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

        x_resnet1 = self.reslayer1(x2_active)
        x_resnet2 = self.reslayer2(x_resnet1)

        x3 = self.conv3(x_resnet2)
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

def test():
    #net = HSID(24)
    net = HSIDRes(24)
    print(net)

    data = torch.randn(1, 24, 200, 200)
    data1 = torch.randn(1, 1, 200, 200)

    output = net(data1, data)
    print(output.shape)

#test()

class TwoStageHSID(nn.Module):
    def __init__(self, k=24):
        super(TwoStageHSID, self).__init__()
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

        ##stage two
        self.spatial_feature_3_stage2 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5_stage2 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7_stage2 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3_stage2 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5_stage2 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7_stage2 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu_stage2 = nn.ReLU()
        #self.feature_3_5_7 concat + relu

        #self.feature_all : Concat
        self.conv1_stage2 = nn.Conv2d(120, 60, kernel_size=3, stride=1, padding=1)
        self.conv2_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv3_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv4_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv5_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv6_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv7_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv8_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv9_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        

        self.feature_conv3_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv5_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv7_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
        self.feature_conv9_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
         
        #self.feature_conv_3_5_7_9 = concat
        self.conv10_stage2 = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)



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

        #stage two forward
        x_spatial_stage2 = x_spatial + output
        x_spatial_feature_3_stage2 = self.spatial_feature_3_stage2(x_spatial_stage2)
        x_spatial_feature_5_stage2 = self.spatial_feature_5_stage2(x_spatial_stage2)
        x_spatial_feature_7_stage2 = self.spatial_feature_7_stage2(x_spatial_stage2)

        x_spectral_feature_3_stage2 = self.spectral_feature_3_stage2(x_spectral)
        x_spectral_feature_5_stage2 = self.spectral_feature_5_stage2(x_spectral)
        x_spectral_feature_7_stage2 = self.spectral_feature_7_stage2(x_spectral)

        feature_3_5_7_stage2 = torch.cat((x_spatial_feature_3_stage2, x_spatial_feature_5_stage2, x_spatial_feature_7_stage2), dim=1) #在通道维concat
        feature_3_5_7_stage2 = self.relu(feature_3_5_7_stage2)
        #print('feature_3_5_7 shape =', feature_3_5_7.shape)

        feature_3_5_7_2_stage2 = torch.cat((x_spectral_feature_3_stage2, x_spectral_feature_5_stage2, x_spectral_feature_7_stage2), dim=1) # 在通道维concat
        feature_3_5_7_2_stage2 = self.relu(feature_3_5_7_2_stage2)
        #print('feature_3_5_7_2 shape =', feature_3_5_7_2.shape)

        feature_all_stage2 = torch.cat((feature_3_5_7_stage2, feature_3_5_7_2_stage2), dim=1)
        #print('feature_all shape =', feature_all.shape)

        x1_stage2 = self.conv1(feature_all_stage2)
        x1_active_stage2 = self.relu(x1_stage2)
        x2_stage2 = self.conv2(x1_active_stage2)
        x2_active_stage2 = self.relu(x2_stage2)
        x3_stage2 = self.conv3(x2_active_stage2)
        x3_active_stage2 = self.relu(x3_stage2)

        x4_stage2 = self.conv4(x3_active_stage2)
        x4_active_stage2 = self.relu(x4_stage2)
        x5_stage2 = self.conv5(x4_active_stage2)
        x5_active_stage2 = self.relu(x5_stage2)
        x6_stage2 = self.conv6(x5_active_stage2)
        x6_active_stage2 = self.relu(x6_stage2)

        x7_stage2 = self.conv7(x6_active_stage2)
        x7_active_stage2 = self.relu(x7_stage2)
        x8_stage2 = self.conv8(x7_active_stage2)
        x8_active_stage2 = self.relu(x8_stage2)
        x9_stage2 = self.conv9(x8_active_stage2)
        x9_active_stage2 = self.relu(x9_stage2)

        feature_conv3_stage2 = self.feature_conv3_stage2(x3_active_stage2)
        feature_conv5_stage2 = self.feature_conv5_stage2(x5_active_stage2)
        feature_conv7_stage2 = self.feature_conv7_stage2(x7_active_stage2)
        feature_conv9_stage2 = self.feature_conv9_stage2(x9_active_stage2)

        feature_conv_3_5_7_9_stage2 = torch.cat((feature_conv3_stage2, feature_conv5_stage2, feature_conv7_stage2, feature_conv9_stage2), dim=1)
        
        feature_conv_3_5_7_9_stage2 = self.relu(feature_conv_3_5_7_9_stage2)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        output_stage2 = self.conv10(feature_conv_3_5_7_9_stage2)
                
        return output, output_stage2 # + x_spatial

from AttentionUNet import *

class TwoStageHSIDWithUNet(nn.Module):
    def __init__(self, k=24):
        super(TwoStageHSIDWithUNet, self).__init__()
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

        ##stage two
        self.spatial_feature_3_stage2 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5_stage2 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7_stage2 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3_stage2 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5_stage2 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7_stage2 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu_stage2 = nn.ReLU()
        #self.feature_3_5_7 concat + relu

        #self.feature_all : Concat
        self.conv1_stage2 = nn.Conv2d(120, 60, kernel_size=3, stride=1, padding=1)
        self.conv2_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv3_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv4_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv5_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv6_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv7_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv8_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv9_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        

        self.feature_conv3_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv5_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv7_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
        self.feature_conv9_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
         
        #self.feature_conv_3_5_7_9 = concat
        self.conv10_stage2 = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)
        self.unet = AttentionUNet(60, 3, 4, 48, False)


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


        unet_res = self.unet(x1_active)

        output = self.conv10(unet_res)

        #stage two forward
        x_spatial_stage2 = x_spatial + output
        x_spatial_feature_3_stage2 = self.spatial_feature_3_stage2(x_spatial_stage2)
        x_spatial_feature_5_stage2 = self.spatial_feature_5_stage2(x_spatial_stage2)
        x_spatial_feature_7_stage2 = self.spatial_feature_7_stage2(x_spatial_stage2)

        x_spectral_feature_3_stage2 = self.spectral_feature_3_stage2(x_spectral)
        x_spectral_feature_5_stage2 = self.spectral_feature_5_stage2(x_spectral)
        x_spectral_feature_7_stage2 = self.spectral_feature_7_stage2(x_spectral)

        feature_3_5_7_stage2 = torch.cat((x_spatial_feature_3_stage2, x_spatial_feature_5_stage2, x_spatial_feature_7_stage2), dim=1) #在通道维concat
        feature_3_5_7_stage2 = self.relu(feature_3_5_7_stage2)
        #print('feature_3_5_7 shape =', feature_3_5_7.shape)

        feature_3_5_7_2_stage2 = torch.cat((x_spectral_feature_3_stage2, x_spectral_feature_5_stage2, x_spectral_feature_7_stage2), dim=1) # 在通道维concat
        feature_3_5_7_2_stage2 = self.relu(feature_3_5_7_2_stage2)
        #print('feature_3_5_7_2 shape =', feature_3_5_7_2.shape)

        feature_all_stage2 = torch.cat((feature_3_5_7_stage2, feature_3_5_7_2_stage2), dim=1)
        #print('feature_all shape =', feature_all.shape)

        x1_stage2 = self.conv1(feature_all_stage2)
        x1_active_stage2 = self.relu(x1_stage2)
        x2_stage2 = self.conv2(x1_active_stage2)
        x2_active_stage2 = self.relu(x2_stage2)
        x3_stage2 = self.conv3(x2_active_stage2)
        x3_active_stage2 = self.relu(x3_stage2)

        x4_stage2 = self.conv4(x3_active_stage2)
        x4_active_stage2 = self.relu(x4_stage2)
        x5_stage2 = self.conv5(x4_active_stage2)
        x5_active_stage2 = self.relu(x5_stage2)
        x6_stage2 = self.conv6(x5_active_stage2)
        x6_active_stage2 = self.relu(x6_stage2)

        x7_stage2 = self.conv7(x6_active_stage2)
        x7_active_stage2 = self.relu(x7_stage2)
        x8_stage2 = self.conv8(x7_active_stage2)
        x8_active_stage2 = self.relu(x8_stage2)
        x9_stage2 = self.conv9(x8_active_stage2)
        x9_active_stage2 = self.relu(x9_stage2)

        feature_conv3_stage2 = self.feature_conv3_stage2(x3_active_stage2)
        feature_conv5_stage2 = self.feature_conv5_stage2(x5_active_stage2)
        feature_conv7_stage2 = self.feature_conv7_stage2(x7_active_stage2)
        feature_conv9_stage2 = self.feature_conv9_stage2(x9_active_stage2)

        feature_conv_3_5_7_9_stage2 = torch.cat((feature_conv3_stage2, feature_conv5_stage2, feature_conv7_stage2, feature_conv9_stage2), dim=1)
        
        feature_conv_3_5_7_9_stage2 = self.relu(feature_conv_3_5_7_9_stage2)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        output_stage2 = self.conv10(feature_conv_3_5_7_9_stage2)
                
        return output, output_stage2 # + x_spatial

class TwoStageHSID_Conv11(nn.Module):
    def __init__(self, k=24):
        super(TwoStageHSID_Conv11, self).__init__()
        self.spatial_feature_1 = nn.Conv2d(1, 20, kernel_size=1, stride=1, padding=0)
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_1 = nn.Conv2d(k, 20, kernel_size=1, stride=1, padding=0)
        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu

        #self.feature_all : Concat
        self.conv1 = nn.Conv2d(160, 60, kernel_size=3, stride=1, padding=1)
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

        ##stage two
        self.spatial_feature_1_stage2 = nn.Conv2d(1, 20, kernel_size=1, stride=1, padding=0)
        self.spatial_feature_3_stage2 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5_stage2 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7_stage2 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_1_stage2 = nn.Conv2d(k, 20, kernel_size=1, stride=1, padding=0)
        self.spectral_feature_3_stage2 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5_stage2 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7_stage2 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu_stage2 = nn.ReLU()
        #self.feature_3_5_7 concat + relu

        #self.feature_all : Concat
        self.conv1_stage2 = nn.Conv2d(160, 60, kernel_size=3, stride=1, padding=1)
        self.conv2_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv3_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv4_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv5_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv6_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv7_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv8_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv9_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        

        self.feature_conv3_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv5_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv7_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
        self.feature_conv9_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
         
        #self.feature_conv_3_5_7_9 = concat
        self.conv10_stage2 = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)



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

        #stage two forward
        x_spatial_stage2 = x_spatial + output
        x_spatial_feature_1_stage2 = self.spatial_feature_1_stage2(x_spatial_stage2)
        x_spatial_feature_3_stage2 = self.spatial_feature_3_stage2(x_spatial_stage2)
        x_spatial_feature_5_stage2 = self.spatial_feature_5_stage2(x_spatial_stage2)
        x_spatial_feature_7_stage2 = self.spatial_feature_7_stage2(x_spatial_stage2)

        x_spectral_feature_1_stage2 = self.spectral_feature_1_stage2(x_spectral)
        x_spectral_feature_3_stage2 = self.spectral_feature_3_stage2(x_spectral)
        x_spectral_feature_5_stage2 = self.spectral_feature_5_stage2(x_spectral)
        x_spectral_feature_7_stage2 = self.spectral_feature_7_stage2(x_spectral)

        feature_3_5_7_stage2 = torch.cat((x_spatial_feature_1_stage2, x_spatial_feature_3_stage2, x_spatial_feature_5_stage2, x_spatial_feature_7_stage2), dim=1) #在通道维concat
        feature_3_5_7_stage2 = self.relu(feature_3_5_7_stage2)
        #print('feature_3_5_7 shape =', feature_3_5_7.shape)

        feature_3_5_7_2_stage2 = torch.cat((x_spectral_feature_1_stage2, x_spectral_feature_3_stage2, x_spectral_feature_5_stage2, x_spectral_feature_7_stage2), dim=1) # 在通道维concat
        feature_3_5_7_2_stage2 = self.relu(feature_3_5_7_2_stage2)
        #print('feature_3_5_7_2 shape =', feature_3_5_7_2.shape)

        feature_all_stage2 = torch.cat((feature_3_5_7_stage2, feature_3_5_7_2_stage2), dim=1)
        #print('feature_all shape =', feature_all.shape)

        x1_stage2 = self.conv1(feature_all_stage2)
        x1_active_stage2 = self.relu(x1_stage2)
        x2_stage2 = self.conv2(x1_active_stage2)
        x2_active_stage2 = self.relu(x2_stage2)
        x3_stage2 = self.conv3(x2_active_stage2)
        x3_active_stage2 = self.relu(x3_stage2)

        x4_stage2 = self.conv4(x3_active_stage2)
        x4_active_stage2 = self.relu(x4_stage2)
        x5_stage2 = self.conv5(x4_active_stage2)
        x5_active_stage2 = self.relu(x5_stage2)
        x6_stage2 = self.conv6(x5_active_stage2)
        x6_active_stage2 = self.relu(x6_stage2)

        x7_stage2 = self.conv7(x6_active_stage2)
        x7_active_stage2 = self.relu(x7_stage2)
        x8_stage2 = self.conv8(x7_active_stage2)
        x8_active_stage2 = self.relu(x8_stage2)
        x9_stage2 = self.conv9(x8_active_stage2)
        x9_active_stage2 = self.relu(x9_stage2)

        feature_conv3_stage2 = self.feature_conv3_stage2(x3_active_stage2)
        feature_conv5_stage2 = self.feature_conv5_stage2(x5_active_stage2)
        feature_conv7_stage2 = self.feature_conv7_stage2(x7_active_stage2)
        feature_conv9_stage2 = self.feature_conv9_stage2(x9_active_stage2)

        feature_conv_3_5_7_9_stage2 = torch.cat((feature_conv3_stage2, feature_conv5_stage2, feature_conv7_stage2, feature_conv9_stage2), dim=1)
        
        feature_conv_3_5_7_9_stage2 = self.relu(feature_conv_3_5_7_9_stage2)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        output_stage2 = self.conv10(feature_conv_3_5_7_9_stage2)
                
        return output, output_stage2 # + x_spatial

class HSID_1x3(nn.Module):
    def __init__(self, k=24):
        super(HSID_1x3, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=(1,3), stride=1, padding=(0,1))
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=(1,5), stride=1, padding=(0,2))
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=(1,7), stride=1, padding=(0,3))

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=(1,3), stride=1, padding=(0,1))
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=(1,5), stride=1, padding=(0,2))
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=(1,7), stride=1, padding=(0,3))

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

class TwoStageHSID_1x3(nn.Module):
    def __init__(self, k=24):
        super(TwoStageHSID_1x3, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=(1,3), stride=1, padding=(0,1))
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=(1,5), stride=1, padding=(0,2))
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=(1,7), stride=1, padding=(0,3))

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=(1,3), stride=1, padding=(0,1))
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=(1,5), stride=1, padding=(0,2))
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=(1,7), stride=1, padding=(0,3))

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

        ##stage two
        self.spatial_feature_3_stage2 = nn.Conv2d(1, 20, kernel_size=(1,3), stride=1, padding=(0,1))
        self.spatial_feature_5_stage2 = nn.Conv2d(1, 20, kernel_size=(1,5), stride=1, padding=(0,2))
        self.spatial_feature_7_stage2 = nn.Conv2d(1, 20, kernel_size=(1,7), stride=1, padding=(0,3))

        self.spectral_feature_3_stage2 = nn.Conv2d(k, 20, kernel_size=(1,3), stride=1, padding=(0,1))
        self.spectral_feature_5_stage2 = nn.Conv2d(k, 20, kernel_size=(1,5), stride=1, padding=(0,2))
        self.spectral_feature_7_stage2 = nn.Conv2d(k, 20, kernel_size=(1,7), stride=1, padding=(0,3))

        #self.feature_3_5_7 concat + relu
        self.relu_stage2 = nn.ReLU()
        #self.feature_3_5_7 concat + relu

        #self.feature_all : Concat
        self.conv1_stage2 = nn.Conv2d(120, 60, kernel_size=3, stride=1, padding=1)
        self.conv2_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv3_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv4_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv5_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv6_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv7_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)
        self.conv8_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        
        self.conv9_stage2 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1)        

        self.feature_conv3_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv5_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv7_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
        self.feature_conv9_stage2 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
         
        #self.feature_conv_3_5_7_9 = concat
        self.conv10_stage2 = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)



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

        #stage two forward
        x_spatial_stage2 = x_spatial + output
        x_spatial_feature_3_stage2 = self.spatial_feature_3_stage2(x_spatial_stage2)
        x_spatial_feature_5_stage2 = self.spatial_feature_5_stage2(x_spatial_stage2)
        x_spatial_feature_7_stage2 = self.spatial_feature_7_stage2(x_spatial_stage2)

        x_spectral_feature_3_stage2 = self.spectral_feature_3_stage2(x_spectral)
        x_spectral_feature_5_stage2 = self.spectral_feature_5_stage2(x_spectral)
        x_spectral_feature_7_stage2 = self.spectral_feature_7_stage2(x_spectral)

        feature_3_5_7_stage2 = torch.cat((x_spatial_feature_3_stage2, x_spatial_feature_5_stage2, x_spatial_feature_7_stage2), dim=1) #在通道维concat
        feature_3_5_7_stage2 = self.relu(feature_3_5_7_stage2)
        #print('feature_3_5_7 shape =', feature_3_5_7.shape)

        feature_3_5_7_2_stage2 = torch.cat((x_spectral_feature_3_stage2, x_spectral_feature_5_stage2, x_spectral_feature_7_stage2), dim=1) # 在通道维concat
        feature_3_5_7_2_stage2 = self.relu(feature_3_5_7_2_stage2)
        #print('feature_3_5_7_2 shape =', feature_3_5_7_2.shape)

        feature_all_stage2 = torch.cat((feature_3_5_7_stage2, feature_3_5_7_2_stage2), dim=1)
        #print('feature_all shape =', feature_all.shape)

        x1_stage2 = self.conv1(feature_all_stage2)
        x1_active_stage2 = self.relu(x1_stage2)
        x2_stage2 = self.conv2(x1_active_stage2)
        x2_active_stage2 = self.relu(x2_stage2)
        x3_stage2 = self.conv3(x2_active_stage2)
        x3_active_stage2 = self.relu(x3_stage2)

        x4_stage2 = self.conv4(x3_active_stage2)
        x4_active_stage2 = self.relu(x4_stage2)
        x5_stage2 = self.conv5(x4_active_stage2)
        x5_active_stage2 = self.relu(x5_stage2)
        x6_stage2 = self.conv6(x5_active_stage2)
        x6_active_stage2 = self.relu(x6_stage2)

        x7_stage2 = self.conv7(x6_active_stage2)
        x7_active_stage2 = self.relu(x7_stage2)
        x8_stage2 = self.conv8(x7_active_stage2)
        x8_active_stage2 = self.relu(x8_stage2)
        x9_stage2 = self.conv9(x8_active_stage2)
        x9_active_stage2 = self.relu(x9_stage2)

        feature_conv3_stage2 = self.feature_conv3_stage2(x3_active_stage2)
        feature_conv5_stage2 = self.feature_conv5_stage2(x5_active_stage2)
        feature_conv7_stage2 = self.feature_conv7_stage2(x7_active_stage2)
        feature_conv9_stage2 = self.feature_conv9_stage2(x9_active_stage2)

        feature_conv_3_5_7_9_stage2 = torch.cat((feature_conv3_stage2, feature_conv5_stage2, feature_conv7_stage2, feature_conv9_stage2), dim=1)
        
        feature_conv_3_5_7_9_stage2 = self.relu(feature_conv_3_5_7_9_stage2)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        output_stage2 = self.conv10(feature_conv_3_5_7_9_stage2)
                
        return output, output_stage2 # + x_spatial