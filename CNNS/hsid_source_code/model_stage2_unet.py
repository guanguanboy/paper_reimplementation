import torch
import torch.nn as nn
import torch.nn.functional as F

"""
RelightNet,输入和输出的大小是一样的

"""
class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()

        self.relu         = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(60, channel, kernel_size,
                                      padding=1, padding_mode='replicate')

        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')

        self.net2_deconv1_1= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')
        self.net2_deconv1_2= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')
        self.net2_deconv1_3= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')

        self.net2_fusion = nn.Conv2d(channel*3, channel, kernel_size=1,
                                     padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

    def forward(self, input_img):#input_L 是1通道的，表示的是ilummination map, input_R是反射率图像。
        out0      = self.net2_conv0_1(input_img)
        out1      = self.relu(self.net2_conv1_1(out0))
        out2      = self.relu(self.net2_conv1_2(out1))
        out3      = self.relu(self.net2_conv1_3(out2))

        out3_up   = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        deconv1   = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up= F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2   = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up= F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3   = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))

        #将三个反卷积之后的特征cat起来，因为deconv1_rs和deconv2_rs大小与deconv3不同。
        #所以需先对deconv1_rs和deconv2_rs进行上采样
        deconv1_rs= F.interpolate(deconv1, size=(input_img.size()[2], input_img.size()[3]))
        deconv2_rs= F.interpolate(deconv2, size=(input_img.size()[2], input_img.size()[3]))
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        
        feats_fus = self.net2_fusion(feats_all)
        output    = self.net2_output(feats_fus)
        return output

def conv_relu(inchannels, outchannels):
    layer = [nn.Conv2d(inchannels, outchannels, 3, 1, 1), nn.ReLU(inplace=True)]
    return layer

class HSIDDenseNetTwoStageUNet(nn.Module):
    def __init__(self, k=24):
        super(HSIDDenseNetTwoStageUNet, self).__init__()
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

        self.relight_net = RelightNet()

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

        residual = self.conv10(feature_conv_3_5_7_9)
        refined_residual = self.relight_net(feature_conv_3_5_7_9)
        #refined_residual = self.conv10_stage2(refined_features)

        return residual, refined_residual # + x_spatial


def relight_net_test():
    net = RelightNet()

    intput_tensor_L = torch.randn(1, 1, 20, 20)
    intput_tensor_R = torch.randn(1, 3, 20, 20)

    output = net(intput_tensor_L)
    print(output.shape)

def two_stage_test():
    net = HSIDDenseNetTwoStageUNet(36)
    intput_tensor_L = torch.randn(1, 1, 20, 20)
    spectal_tensor = torch.randn(1, 36, 20, 20)
    output,stage2_residual = net(intput_tensor_L, spectal_tensor)
    print(output.shape, stage2_residual.shape)

#if __name__ == "__main__":
    #relight_net_test()
    #two_stage_test()

