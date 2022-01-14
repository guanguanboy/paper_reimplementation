
import torch

import torch.nn as nn


import torch.nn.functional as F
import torch.nn.init as init


class HSIDCNN(nn.Module):
    def __init__(self):
        super(HSIDCNN, self).__init__()

        def conv_relu(inchannels, outchannels):
            layer = [nn.Conv2d(inchannels, outchannels, 3, 1, 1), nn.ReLU(inplace=True)]
            return layer

        #self.f3_1 = nn.Conv2d(2, 20, 3, 1, 1)
        #self.f5_1 = nn.Conv2d(2, 20, 5, 1, 2)
        #self.f7_1 = nn.Conv2d(2, 20, 7, 1, 3)
        self.f3_1 = nn.Conv2d(1, 20, 3, 1, 1)
        self.f5_1 = nn.Conv2d(1, 20, 5, 1, 2)
        self.f7_1 = nn.Conv2d(1, 20, 7, 1, 3)

        self.f3_2 = nn.Conv3d(1, 20, (36, 3, 3), 1, (0, 1, 1))
        self.f5_2 = nn.Conv3d(1, 20, (36, 5, 5), 1, (0, 2, 2))
        self.f7_2 = nn.Conv3d(1, 20, (36, 7, 7), 1, (0, 3, 3))

        self.conv1 = nn.Sequential(*conv_relu(120, 60), )
        self.conv2 = nn.Sequential(*conv_relu(60, 60), )
        self.conv3 = nn.Sequential(*conv_relu(60, 60), )
        self.conv4 = nn.Sequential(*conv_relu(60, 60), )
        self.conv5 = nn.Sequential(*conv_relu(60, 60), )
        self.conv6 = nn.Sequential(*conv_relu(60, 60), )
        self.conv7 = nn.Sequential(*conv_relu(60, 60), )
        self.conv8 = nn.Sequential(*conv_relu(60, 60), )
        self.conv9 = nn.Sequential(*conv_relu(60, 60), )
        self.conv10 = nn.Conv2d(240, 1, 3, 1, 1)

    def forward(self, x, y):


        f3_1 = self.f3_1(x)
        f5_1 = self.f5_1(x)
        f7_1 = self.f7_1(x)


        f3_2 = self.f3_2(y)
        f5_2 = self.f5_2(y)
        f7_2 = self.f7_2(y)



        out1 = F.relu(torch.cat((f3_1, f5_1, f7_1), dim=1)) # 12 60 20 20
        out2 = F.relu(torch.cat((f3_2, f5_2, f7_2), dim=1)).squeeze(2) ## 12 60 20 20
        out3 = torch.cat((out1, out2), dim=1) ## 12 120 20 20

        conv1 = self.conv1(out3)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)
        conv9 = self.conv9(conv8)

        concat = torch.cat((conv3, conv5, conv7, conv9), dim=1)

        out_final = self.conv10(concat)

        return x+ out_final

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

