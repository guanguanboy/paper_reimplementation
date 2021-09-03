import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from hsidataset import HsiCubicTrainDataset
import numpy as np
from metrics import PSNR, SSIM, SAM
from hsidataset import HsiCubicTestDataset,HsiCubicLowlightTestDataset
import scipy.io as scio
from helper.helper_utils import init_params, get_summary_writer
recon_criterion = nn.L1Loss() 
enlighten_loss = nn.MSELoss()

DENOISE_PHASE = "Denoise"
ENLIGHTEN_PHASE = "Enlighten"

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

        residual = self.conv10(feature_conv_3_5_7_9)

        return residual # + x_spatial

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

class RelightNet(nn.Module):
    def __init__(self):
        super(RelightNet, self).__init__()

        self.expand_channel = transition_block(1, 20)
        # `num_channels`为当前的通道数
        num_channels = 20
        growth_rate = 10
        num_convs_in_dense_blocks = [3, 3] #添加了两个dense_block
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            # 上一个稠密块的输出通道数
            num_channels += num_convs * growth_rate
            # 在稠密块之间添加一个转换层，使通道数量减半
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

        self.desenblock = nn.Sequential(*blks,
        nn.BatchNorm2d(num_channels), nn.ReLU(),transition_block(num_channels, num_channels // 2))

        self.squeeze_channel = transition_block(num_channels//2, 1)

    def forward(self, net1_restored):
        identity = net1_restored
        x = self.expand_channel(net1_restored)
        x = self.desenblock(x)
        net2_residual = self.squeeze_channel(x)

        return net2_residual

def relight_net_test():
    net = RelightNet()

    input_tensor = torch.randn(2, 1, 200, 200)
    output = net(input_tensor)
    print(output.shape)



class EnlightenHyperSpectralNet(nn.Module):
    def __init__(self, k=24):
        super(EnlightenHyperSpectralNet,self).__init__()
        self.hsid = HSIDRefactored(k)
        self.relight_net = RelightNet()

        self.train_phase = ENLIGHTEN_PHASE

    def forward(self, x_spatial, x_spectral):
        hsid_residual = self.hsid(x_spatial, x_spectral)

        hsid_restored = x_spatial + hsid_residual
        enlightened_residual = self.relight_net(hsid_restored)

        return hsid_residual, enlightened_residual

    def train_model(self, train_data_dir, test_data_dir, test_label_dir, batch_size,
            epoch_num, init_lr, ckpt_dir, device, display_step, train_phase):
        
        self.train_phase = train_phase
        print("Start training for phase %s" % (self.train_phase))

        #准备数据
        train_set = HsiCubicTrainDataset(train_data_dir) #'./data/train_lowlight/'
        print('total training example:', len(train_set))

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

        #加载测试label数据
        mat_src_path = test_label_dir
        test_label_hsi = scio.loadmat(mat_src_path)['label']

        #加载测试数据
        batch_size = 1
        #test_data_dir = './data/test_lowlight/cubic/'
        test_set = HsiCubicLowlightTestDataset(test_data_dir)
        test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

        #准备优化器
        self.hsid_optimizer = optim.Adam(self.hsid.parameters(), lr=init_lr)
        self.hsid_scheduler = MultiStepLR(self.hsid_optimizer, milestones=[40,60,80], gamma=0.1)

        self.enlighter_optimizer = optim.Adam(self.relight_net.parameters(), lr=init_lr)
        self.enlighter_scheduler = MultiStepLR(self.enlighter_optimizer, milestones=[40,60,80], gamma=0.1)


        #加载模型
        self.init_model(ckpt_dir)
        self.to(device)
        if self.train_phase == ENLIGHTEN_PHASE:
            for param in self.hsid.parameters():
                param.reguires_grad = False
        elif self.train_phase == DENOISE_PHASE:
            for param in self.relight_net.parameters():
                param.reguires_grad = False
        #按照epoch进行训练
        global tb_writer
        tb_writer = get_summary_writer(log_dir='logs')

        gen_epoch_loss_list = []

        cur_step = 0

        self.best_psnr = 0
        self.best_epoch = 0
        self.best_iter = 0
        start_epoch = 1
        num_epoch = epoch_num

        for epoch in range(start_epoch, num_epoch+1):
            epoch_start_time = time.time()
            if self.train_phase == DENOISE_PHASE:
                scheduler = self.hsid_scheduler
            else:
                scheduler = self.enlighter_scheduler

            scheduler.step()
            print(epoch, 'lr={:.6f}'.format(scheduler.get_last_lr()[0]))
            print(scheduler.get_lr())
            
            gen_epoch_loss = 0

            self.train()
            #for batch_idx, (noisy, label) in enumerate([first_batch] * 300):
            for batch_idx, (noisy, cubic, label) in enumerate(train_loader):
                #print('batch_idx=', batch_idx)
                noisy = noisy.to(device)
                label = label.to(device)
                cubic = cubic.to(device)

                #denoised_img = net(noisy, cubic)
                #loss = loss_fuction(denoised_img, label)

                hsid_residual, enlighter_residual = self.forward(noisy, cubic)

                if train_phase == DENOISE_PHASE:
                    loss = recon_criterion(hsid_residual, label-noisy)
                    self.hsid_optimizer.zero_grad()
                    loss.backward() # calcu gradient
                    self.hsid_optimizer.step() # update parameter

                else:
                    hsid_restored = noisy + hsid_residual.detach()
                    loss = enlighten_loss(enlighter_residual, label-hsid_restored)
                    self.enlighter_optimizer.zero_grad()
                    loss.backward() # calcu gradient
                    self.enlighter_optimizer.step() # update parameter

                gen_epoch_loss += loss.item()

                if cur_step % display_step == 0:
                    if cur_step > 0:
                        print(f"Epoch {epoch}: Step {cur_step}: Batch_idx {batch_idx}: MSE loss: {loss.item()}")
                    else:
                        print("Pretrained initial state")

                tb_writer.add_scalar("MSE loss", loss.item(), cur_step)

                #step ++,每一次循环，每一个batch的处理，叫做一个step
                cur_step += 1


            gen_epoch_loss_list.append(gen_epoch_loss)
            tb_writer.add_scalar("mse epoch loss", gen_epoch_loss, epoch)

            ##保存模型
            self.save_epoch_model(epoch, ckpt_dir)

            ##对模型进行评价
            # def evaluate(self, epoch, test_dataloader, test_label_hsi, device, ckpt_dir, cur_step):
            self.evaluate_model(epoch=epoch, test_dataloader=test_dataloader, test_label_hsi=test_label_hsi,
            device=device, ckpt_dir=ckpt_dir, cur_step=cur_step)



        tb_writer.close()


    def load_model(self, ckpt_dir, device):
        load_hsid_dir = ckpt_dir + '/' + DENOISE_PHASE + '/'
        if os.path.exists(load_hsid_dir):
            ckpt_dict = torch.load(load_hsid_dir + "enlighten_hyper_hsid_best.pth", map_location=device)
            self.hsid.load_state_dict(ckpt_dict['hsid'])
            
        load_enlighter_dir = ckpt_dir + '/' + ENLIGHTEN_PHASE + '/'
        if os.path.exists(load_enlighter_dir):
            ckpt_dict = torch.load(load_enlighter_dir + "enlighten_hyper_enlighter_best.pth", map_location=device)
            self.relight_net.load_state_dict(ckpt_dict['enlighter'])        

    def init_model(self, ckpt_dir):
        if self.train_phase == DENOISE_PHASE:
            init_params(self.hsid)
        elif self.train_phase ==ENLIGHTEN_PHASE:
            hsid_load_dir = ckpt_dir + '/' + DENOISE_PHASE + '/'
            ckpt_dict = torch.load(hsid_load_dir + "enlighten_hyper_hsid_best.pth")
            self.hsid.load_state_dict(ckpt_dict['hsid'])
            init_params(self.relight_net)

    def save_epoch_model(self, epoch, ckpt_dir):

        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if self.train_phase == DENOISE_PHASE:
            save_name = save_dir + '/enlighten_hyper_hsid' + str(epoch) + '.pth'
            torch.save({
            'hsid': self.hsid.state_dict(),
            'hsid_opt': self.hsid.state_dict(),}, save_name)
        elif self.train_phase ==ENLIGHTEN_PHASE:
            save_name = save_dir + '/enlighten_hyper_enlighter' + str(epoch) + '.pth'
            torch.save({
            'enlighter': self.hsid.state_dict(),
            'enlighter_opt': self.hsid.state_dict(),}, save_name)

    def save_best_model(self, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == DENOISE_PHASE:
            save_name = save_dir + '/enlighten_hyper_hsid_best.pth'
            torch.save({
            'hsid': self.hsid.state_dict(),
            'hsid_opt': self.hsid.state_dict(),}, save_name)
        elif self.train_phase == ENLIGHTEN_PHASE:
            save_name = save_dir + '/enlighten_hyper_enlighter_best.pth'
            torch.save({
            'enlighter': self.relight_net.state_dict(),
            'enlighter_opt': self.relight_net.state_dict(),}, save_name)

    def evaluate_model(self, epoch, test_dataloader, test_label_hsi, device, ckpt_dir, cur_step):

        batch_size, channel, width, height = next(iter(test_dataloader))[0].shape
        
        band_num = len(test_dataloader)
        denoised_hsi = np.zeros((width, height, band_num))

        #测试代码
        self.eval()
        for batch_idx, (noisy_test, cubic_test, label_test) in enumerate(test_dataloader):
            noisy_test = noisy_test.type(torch.FloatTensor)
            label_test = label_test.type(torch.FloatTensor)
            cubic_test = cubic_test.type(torch.FloatTensor)

            noisy_test = noisy_test.to(device)
            label_test = label_test.to(device)
            cubic_test = cubic_test.to(device)

            with torch.no_grad():

                hsid_residual, enlightened_residual = self.forward(noisy_test, cubic_test)
                if self.train_phase == DENOISE_PHASE:
                    denoised_band = noisy_test + hsid_residual
                elif self.train_phase == ENLIGHTEN_PHASE:
                    denoised_band = noisy_test + hsid_residual + enlightened_residual
                
                denoised_band_numpy = denoised_band.cpu().numpy().astype(np.float32)
                denoised_band_numpy = np.squeeze(denoised_band_numpy)

                denoised_hsi[:,:,batch_idx] = denoised_band_numpy

                if batch_idx == 49:
                    if self.train_phase == DENOISE_PHASE: 
                        residual_squeezed = torch.squeeze(hsid_residual, axis=0)
                    elif self.train_phase == ENLIGHTEN_PHASE:
                        residual_squeezed = torch.squeeze(hsid_residual+enlightened_residual, axis=0)

                    denoised_band_squeezed = torch.squeeze(denoised_band, axis=0) 
                    label_test_squeezed = torch.squeeze(label_test,axis=0)
                    noisy_test_squeezed = torch.squeeze(noisy_test,axis=0)
                    tb_writer.add_image(f"images/{epoch}_restored", denoised_band_squeezed, 1, dataformats='CHW')
                    tb_writer.add_image(f"images/{epoch}_residual", residual_squeezed, 1, dataformats='CHW')
                    tb_writer.add_image(f"images/{epoch}_label", label_test_squeezed, 1, dataformats='CHW')
                    tb_writer.add_image(f"images/{epoch}_noisy", noisy_test_squeezed, 1, dataformats='CHW')

        psnr = PSNR(denoised_hsi, test_label_hsi)
        ssim = SSIM(denoised_hsi, test_label_hsi)
        sam = SAM(denoised_hsi, test_label_hsi)

        #计算pnsr和ssim
        print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(psnr, ssim, sam)) 
        tb_writer.add_scalars("validation metrics", {'average PSNR':psnr,
                        'average SSIM':ssim,
                        'avarage SAM': sam}, epoch) #通过这个我就可以看到，那个epoch的性能是最好的

        #保存best模型
        if psnr > self.best_psnr:
            self.best_psnr = psnr
            self.best_epoch = epoch
            self.best_iter = cur_step
            self.save_best_model(ckpt_dir)

        print("[epoch %d it %d PSNR: %.4f --- best_epoch %d best_iter %d Best_PSNR %.4f]" % (epoch, cur_step, psnr, self.best_epoch, self.best_iter, self.best_psnr))

        #print("------------------------------------------------------------------")
        #print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, gen_epoch_loss, scheduler.get_lr()[0]))
        #print("------------------------------------------------------------------")


if __name__ == '__main__':
    relight_net_test()