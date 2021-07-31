
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

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class MultiStageHSIDUpscale(nn.Module):
    def __init__(self, k=24):
        super(MultiStageHSIDUpscale, self).__init__()

        self.stage1_hsid = HSIDRefactored(k)
        self.stage1_upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.stage2_hsid = HSIDRefactored(k)
        self.stage2_upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.stage3_hsid = HSIDRefactored(k)

    def forward(self, x_spatial, x_spectral):
        
        x_spatial_half = F.interpolate(x_spatial, scale_factor=0.5, mode="bilinear")
        x_spectral_half = F.interpolate(x_spectral, scale_factor=0.5, mode="bilinear")

        x_spatial_quarter = F.interpolate(x_spatial, scale_factor=0.25, mode="bilinear")
        x_spectral_quarter = F.interpolate(x_spectral, scale_factor=0.25, mode="bilinear")

        stage1_residual =  self.stage1_hsid(x_spatial_quarter, x_spectral_quarter)
        stage1_restored = stage1_residual + x_spatial_quarter

        stage1_res = self.stage1_upscale(stage1_restored)

        stage2_residual = self.stage2_hsid(stage1_res, x_spectral_half)
        stage2_restored = stage2_residual + x_spatial_half

        stage2_res = self.stage2_upscale(stage2_restored)
        
        stage3_residual = self.stage3_hsid(stage2_res, x_spectral)
        stage3_restored = stage3_residual + x_spatial

        return stage3_restored, stage2_restored, stage1_restored