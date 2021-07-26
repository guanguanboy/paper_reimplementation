
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_relu(inchannels, outchannels):
    layer = [nn.Conv2d(inchannels, outchannels, 3, 1, 1), nn.ReLU(inplace=True)]
    return layer

class ShallowFeatureExtractor(nn.Module):
    def __init__(self, k=24):
        super(ShallowFeatureExtractor, self).__init__()
        self.spatial_feature_3 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.spatial_feature_5 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.spatial_feature_7 = nn.Conv2d(1, 20, kernel_size=7, stride=1, padding=3)

        self.spectral_feature_3 = nn.Conv2d(k, 20, kernel_size=3, stride=1, padding=1)
        self.spectral_feature_5 = nn.Conv2d(k, 20, kernel_size=5, stride=1, padding=2)
        self.spectral_feature_7 = nn.Conv2d(k, 20, kernel_size=7, stride=1, padding=3)

        #self.feature_3_5_7 concat + relu
        self.relu = nn.ReLU()
        #self.feature_3_5_7 concat + relu

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
        # 
        return feature_all

class DeepFeatureExtractor(nn.Module):
    def __init__(self):
        super(DeepFeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(*conv_relu(120, 60))
        self.conv2_3 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv4_5 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv6_7 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))
        self.conv8_9 = nn.Sequential(*conv_relu(60, 60), *conv_relu(60,60))     

        self.feature_conv3 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv5 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1) 
        self.feature_conv7 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
        self.feature_conv9 = nn.Conv2d(60, 15, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()

        #self.feature_conv_3_5_7_9 = concat
        #self.conv10 = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, shallow_feat):
        x1 = self.conv1(shallow_feat)
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

        #output = self.conv10(feature_conv_3_5_7_9)
        deep_feature = feature_conv_3_5_7_9
        return deep_feature

class MultiStageHSID(nn.Module):
    def __init__(self, k=24):
        super(MultiStageHSID, self).__init__()
        self.stage1_shallow_feat = ShallowFeatureExtractor(k)
        self.stage2_shallow_feat = ShallowFeatureExtractor(k)
        self.stage3_shallow_feat = ShallowFeatureExtractor(k)

        self.stage1_deep_feat = DeepFeatureExtractor()
        self.stage2_deep_feat = DeepFeatureExtractor()
        self.stage3_deep_feat = DeepFeatureExtractor()

        self.stage1_reconstruct_residual = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)
        self.stage2_reconstruct_residual = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)
        self.stage3_reconstruct_residual = nn.Conv2d(60, 1, kernel_size=3, stride=1, padding=1)

        self.concat12 = nn.Conv2d(180, 120, kernel_size=1)
        self.concat23 = nn.Conv2d(180, 120, kernel_size=1)

    def forward(self, x_spatial, x_spectral):
        #第3 stage中输入图像分辨率的大小
        image_height = x_spatial.shape[2]
        image_width = x_spatial.shape[3]

        spectral_height = x_spectral.shape[2]
        spectral_width = x_spectral.shape[3]

        #第2 stage中输入的两个patch，将原图像分为上下两部分
        stage2_top_spatial = x_spatial[:,:,0:int(image_height/2),:]
        stage2_bot_spatial = x_spatial[:,:,int(image_height/2):image_height,:]

        stage2_top_spectral = x_spectral[:,:,0:int(spectral_height/2),:]
        stage2_bot_spectral = x_spectral[:,:,int(spectral_height/2):spectral_height,:]

        #第1 stage输入的四个patch，将原图像分成4份
        stage1_top_left_spatial = stage2_top_spatial[:,:,:,0:int(image_width/2)]
        stage1_top_right_spatial = stage2_top_spatial[:,:,:,int(image_width/2):image_width]
        stage1_bot_left_spatial = stage2_bot_spatial[:,:,:,0:int(image_width/2)]
        stage1_bot_right_spatial = stage2_bot_spatial[:,:,:,int(image_width/2):image_width]

        stage1_top_left_spectral = stage2_top_spectral[:,:,:,0:int(spectral_width/2)]
        stage1_top_right_spectral = stage2_top_spectral[:,:,:,int(spectral_width/2):spectral_width]
        stage1_bot_left_spectral = stage2_bot_spectral[:,:,:,0:int(spectral_width/2)]
        stage1_bot_right_spectral = stage2_bot_spectral[:,:,:,int(spectral_width/2):spectral_width]

        ## stage 1 计算浅层特征
        stage1_top_left_feat = self.stage1_shallow_feat(stage1_top_left_spatial, stage1_top_left_spectral)
        stage1_top_right_feat = self.stage1_shallow_feat(stage1_top_right_spatial, stage1_top_right_spectral)
        stage1_bot_left_feat = self.stage1_shallow_feat(stage1_bot_left_spatial, stage1_bot_left_spectral)
        stage1_bot_right_feat = self.stage1_shallow_feat(stage1_bot_right_spatial, stage1_bot_right_spectral)

        ## stage 1 计算深层特征
        stage1_top_left_deep_feat = self.stage1_deep_feat(stage1_top_left_feat)
        stage1_top_right_deep_feat = self.stage1_deep_feat(stage1_top_right_feat)
        stage1_bot_left_deep_feat = self.stage1_deep_feat(stage1_bot_left_feat)
        stage1_bot_right_deep_feat = self.stage1_deep_feat(stage1_bot_right_feat)

        ## stage 1 重建残差residual
        stage1_top_left_residual = self.stage1_reconstruct_residual(stage1_top_left_deep_feat)
        stage1_top_right_residual = self.stage1_reconstruct_residual(stage1_top_right_deep_feat)
        stage1_bot_left_residual = self.stage1_reconstruct_residual(stage1_bot_left_deep_feat)
        stage1_bot_right_residual = self.stage1_reconstruct_residual(stage1_bot_right_deep_feat)


        ## 计算stage1 输出残差图像
        stage1_top_residual = torch.cat([stage1_top_left_residual, stage1_top_right_residual], 3)
        stage1_bot_residual = torch.cat([stage1_bot_left_residual, stage1_bot_right_residual], 3)
        stage1_residual = torch.cat([stage1_top_residual, stage1_bot_residual],2)

        ## stage 1 复原图像
        stage1_top_left_restored = stage1_top_left_residual + stage1_top_left_spatial
        stage1_top_right_restored = stage1_top_right_residual + stage1_top_right_spatial
        stage1_bot_left_restored = stage1_bot_left_residual + stage1_bot_left_spatial
        stage1_bot_right_restored = stage1_bot_right_residual + stage1_bot_right_spatial

 
        ## stage2 计算浅层特征
        stage2_top_feat = self.stage2_shallow_feat(stage2_top_spatial, stage2_top_spectral)
        stage2_bot_feat = self.stage2_shallow_feat(stage2_bot_spatial, stage2_bot_spectral)


        ## stage1 sam特征与stage2浅层特征融合
        stage1_top_deep_feat_fused = torch.cat([stage1_top_left_deep_feat, stage1_top_right_deep_feat],3)
        stage1_bot_deep_feat_fused = torch.cat([stage1_bot_left_deep_feat, stage1_bot_right_deep_feat],3)

        #下面这两个操作需要看一下是否需要在torch.cat之后再增加一个卷积层???? ，需要增加，将通道数在降回来
        stage2_top_cat = self.concat12(torch.cat([stage2_top_feat, stage1_top_deep_feat_fused], 1))
        stage2_bot_cat = self.concat12(torch.cat([stage2_bot_feat, stage1_bot_deep_feat_fused], 1))

        ## stage2 计算深层特征
        stage2_top_deep_feat = self.stage2_deep_feat(stage2_top_cat)
        stage2_bot_deep_feat = self.stage2_deep_feat(stage2_bot_cat)

        ## stage2 重建残差residual
        stage2_top_residual = self.stage2_reconstruct_residual(stage2_top_deep_feat)
        stage2_bot_residual = self.stage2_reconstruct_residual(stage2_bot_deep_feat)

        ##计算stage2 输出残差图像
        stage2_residual = torch.cat([stage2_top_residual, stage2_bot_residual], 2)

        ## stage2 复原图像
        stage2_top_restored = stage2_top_residual + stage2_top_spatial
        stage2_bot_restored = stage2_bot_residual + stage2_bot_spatial

        ## stage3 计算浅层特征
        stage3_shallow_feat = self.stage3_shallow_feat(x_spatial, x_spectral)

        ## 将stage2 sam特征与 stage3 浅层特征进行融合
        stage2_deep_feat_fused = torch.cat([stage2_top_deep_feat, stage2_bot_deep_feat], 2)
        stage3_cat = self.concat23(torch.cat([stage2_deep_feat_fused, stage3_shallow_feat], 1)) ## 在通道维cat

        ## stage3  计算深层特征
        stage3_deep_feat = self.stage3_deep_feat(stage3_cat)

        ## stage3 重建残差
        stage3_residual = self.stage3_reconstruct_residual(stage3_deep_feat)

        ## stage3 复原图像
        stage3_restored = stage3_residual + x_spatial

        return stage3_residual,stage2_residual,stage1_residual

def multi_stage_model_test():
    ms_model = MultiStageHSID(36)

    spectral_data = torch.randn(1, 36, 64, 64)
    spatial_data = torch.randn(1, 1, 64, 64)

    residual = ms_model(spatial_data, spectral_data)

    print(residual[2].shape)


if __name__ == "__main__":
    multi_stage_model_test()




