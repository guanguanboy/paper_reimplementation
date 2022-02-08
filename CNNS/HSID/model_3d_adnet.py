import torch
import torch.nn as nn


def conv_relu(inchannels, outchannels, stride, padding, rate):
    layer = [nn.Conv2d(inchannels, outchannels, 3, stride, padding, dilation=rate), nn.BatchNorm2d(outchannels), nn.LeakyReLU(inplace=True)]
    return layer

def conv_relu_3d(inchannels, outchannels, stride, padding, rate):
    layer = [nn.Conv3d(inchannels, outchannels, (5, 3, 3), stride, padding, dilation=rate), nn.BatchNorm3d(outchannels), nn.LeakyReLU(inplace=True)]
    return layer

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

def conv_3d_large_sacle(inchannels, outchannels, stride, padding, rate):
    layer = [nn.Conv3d(inchannels, outchannels, (5, 5, 5), stride, padding, dilation=rate)]
    return layer

class Model_3D_ADNet(nn.Module):
    def __init__(self, k=24):
        super(Model_3D_ADNet, self).__init__()
        
        #spatial processing branch
        self.conv2d_1 = nn.Sequential(*conv_relu(1, k, 1, 1, 1))
        self.conv2d_2 = nn.Sequential(*conv_relu(k, 32, 1, 1, 1))
        self.conv2d_3 = nn.Sequential(*conv_relu(32, 64, 1, 3, 3))
        self.conv2d_4 = nn.Sequential(*conv_relu(64, k, 1, 5, 5))

        #spectral processing branch
        #self.spectral_feature_3 = nn.Conv3d(1, 20, (k, 3, 3), 1, (0, 1, 1))
        self.conv3d_1 = nn.Sequential(*conv_relu_3d(1, 4, 1, (2,1,1), 1))
        self.conv3d_2 = nn.Sequential(*conv_relu_3d(4, 8, 1, (2,1,1), 1))
        self.conv3d_3 = nn.Sequential(*conv_relu_3d(8, 8, 1, (6, 3,3), 3))
        self.conv3d_4 = nn.Sequential(*conv_relu_3d(8, 1, 1, (10,5,5), 5))
        
        self.pam = PAM_Module(k)
        self.cam = CAM_Module(k)

        #large scale feature extraction
        #由三个3d卷积组成。将输入和输出concate起来，然后使用一个3x3的2d卷积将通道数由2k减为1k。
        self.fe_large_scale_1 = nn.Sequential(*conv_3d_large_sacle(1, 8, 1, (2, 2, 2), 1))
        self.fe_large_scale_2 = nn.Sequential(*conv_3d_large_sacle(8, 8, 1, (2, 2, 2), 1))
        self.fe_large_scale_3 = nn.Sequential(*conv_3d_large_sacle(8, 1, 1, (2, 2, 2), 1))

        self.reduce = nn.Conv2d(k*2, k, 3, 1, 1)

        #multiscale structure
        self.conv2d_multi_1 = nn.Sequential(*conv_relu(k, k, 1, 1, 1))
        self.conv2d_multi_2 = nn.Sequential(*conv_relu(k, k, 1, 3, 3))
        self.conv2d_multi_3 = nn.Sequential(*conv_relu(k, k, 1, 5, 5))
        self.conv2d_multi_4 = nn.Sequential(*conv_relu(k, k, 1, 7, 7))

        self.reduce2 = nn.Sequential(*conv_relu(k*5, 32, 1, 1, 1))

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, x_spatial, x_spectral):
        x_spatial_1 = self.conv2d_1(x_spatial)
        x_spatial_2 = self.conv2d_2(x_spatial_1)
        #print(x_spatial_2.shape)
        x_spatial_3 = self.conv2d_3(x_spatial_2)
        #print(x_spatial_3.shape)
        f_spat = self.conv2d_4(x_spatial_3)
        #The output size of the spatial process branch will be k × H × W,

        x_spectral = x_spectral.unsqueeze(dim=1)
        x_spectral_1 = self.conv3d_1(x_spectral)
        #print(x_spectral_1.shape)

        x_spectral_2 = self.conv3d_2(x_spectral_1)
        #print(x_spectral_2.shape)

        x_spectral_3 = self.conv3d_3(x_spectral_2)
        #print(x_spectral_3.shape)

        x_spectral_4 = self.conv3d_4(x_spectral_3)

        f_spec = x_spectral_4.squeeze(dim=1)

        pam_out = self.pam(f_spat)

        cam_out = self.cam(f_spec)

        f_extr = pam_out + cam_out

        f_extr_unsqueeze = f_extr.unsqueeze(dim=1)
        fe_large_scale_1 = self.fe_large_scale_1(f_extr_unsqueeze)
        fe_large_scale_2 = self.fe_large_scale_2(fe_large_scale_1)
        fe_large_scale_3 = self.fe_large_scale_3(fe_large_scale_2)
        fe_large_scale = fe_large_scale_3.squeeze(dim=1)
        #print('feature extration large scale=', fe_large_scale.shape)

        #concatenate
        feature_cat = torch.cat([fe_large_scale, f_extr], dim=1)

        large_scale_feature_out = self.reduce(feature_cat)

        conv2d_multi_1 =  self.conv2d_multi_1(large_scale_feature_out)
        conv2d_multi_2 =  self.conv2d_multi_2(large_scale_feature_out)
        conv2d_multi_3 =  self.conv2d_multi_3(large_scale_feature_out)
        conv2d_multi_4 =  self.conv2d_multi_4(large_scale_feature_out)

        feature_multi_cat = torch.cat([large_scale_feature_out, conv2d_multi_1, conv2d_multi_2, conv2d_multi_3, conv2d_multi_4], dim=1)
        feature_reduce = self.reduce2(feature_multi_cat)

        output_residual = self.final_conv(feature_reduce)

        return output_residual#output  + x_spatial


def test():
    net = Model_3D_ADNet(24)
    print(net)

    data1 = torch.randn(1, 1, 320, 320)
    data = torch.randn(1, 24, 320, 320)
    """
    pam = PAM_Module(24)
    pam_output = pam(output)
    print('pam output shape:', pam_output.shape)

    cam = CAM_Module(24)
    cam_output = cam(output)
    print('cam output shape:', cam_output.shape)
    """
    output = net(data1, data)
    print(output.shape)


if __name__ == '__main__':
    test()
