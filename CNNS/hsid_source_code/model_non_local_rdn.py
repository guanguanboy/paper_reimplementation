from pickle import FALSE
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdn import *

from NLblock import NONLocalBlock2D



def conv_relu(inchannels, outchannels):
    layer = [nn.Conv2d(inchannels, outchannels, 3, 1, 1), nn.ReLU(inplace=True)]
    return layer
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


class HSIRDNWithNonLocal(nn.Module):
    def __init__(self, k=24):
        super(HSIRDNWithNonLocal, self).__init__()
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

        self.shallow_nonLocal = NONLocalBlock2D(in_channels=60, sub_sample=FALSE, bn_layer=FALSE)

        self.rdn = DenoiseRDN_Custom(60, 20, 4, 3)

        self.deep_nonLocal = NONLocalBlock2D(in_channels=60, sub_sample=FALSE, bn_layer=FALSE)

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

        f0 = self.conv1(feature_all) #x1相当于rdn中的F-1或者f0
        
        f1 = self.shallow_nonLocal(f0)

        feature_rdn = self.rdn(f1)

        feature_conv_3_5_7_9 = self.relu(feature_rdn)
        #print('feature_conv_3_5_7_9 shape=', feature_conv_3_5_7_9.shape)

        nonLocalfeature = self.deep_nonLocal(feature_conv_3_5_7_9)

        output = self.conv10(nonLocalfeature)

        return output

