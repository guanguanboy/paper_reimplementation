
%加噪level 25的noise数据
load('D:\DataSets\hyperspectraldatasets\lowlight_hyperspectral_datasets\lowlight\test\soup_bigcorn_orange_1ms.mat');
figure(1);
title('Low Light Image');
imshow(lowlight(:,:,50));
hold on;
uperleft_x = 131;
uperleft_y = 141;
lowerright_x = 250;
lowerright_y = 260;
width = 120;
height = 120;
drawRectangleImage = rectangle('Position',[uperleft_x,uperleft_y,width,height],'LineWidth',4,'EdgeColor','r');
hold off;
imwrite(lowlight(:,:,50), 'lowlight.png')

figure(2);
imshow(label(:,:,50));
title('Label Image');
imwrite(label(:,:,50), 'label.png')

%加噪去噪后的数据
load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\hsid_origin.mat');
%load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\hsid_rdn_eca_l1_loss_600epoch_patchsize32_best_10_23_k24');
%load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\hsid_rdecab_ablation_l2loss.mat');


figure(3);
imshow(denoised(:,:,50));
title('EnLightened Image');
imwrite(denoised(:,:,50), 'denoised.png')
band_50 = denoised(:,:,50);
imwrite(band_50(131:250, 141:260), 'enlarged_roi_hsid_ablation_l2loss.png')
%imwrite(band_50(131:250, 141:260), 'enlarged_roi_rdecab_ablation_l1loss.png')
%imwrite(band_50(131:250, 141:260), 'enlarged_roi_rdecab_ablation_l2loss.png')

figure(5);
title('residual');
imshow(label(:,:,50) - denoised(:,:,50))
im_label = label;
[w,h, band] = size(im_label);
im_output = denoised;

%% PSNR & SSIM
PSNR=zeros(band, 1);
SSIM=zeros(band, 1);

for i=1:band
 
    [psnr_cur, ssim_cur, ~] = Cal_PSNRSSIM(im_output(:, :, i), im_label(:, :, i), 0, 0);
    PSNR(i,1)=psnr_cur;
    SSIM(i,1)=ssim_cur;
end

[SAM1, SAM2]=SAM(im_label, im_output);
disp(SAM1);

show_band=[57, 27, 17];
%show_band=[30, 20, 10];
figure(4);
subplot(131), imshow(im_label(:, :, show_band));
imwrite(im_label(:, :, show_band), 'sudocolor_label.png')
title('Label Image')

subplot(132), imshow(lowlight(:, :, show_band));
%title(['Noise Level = ', num2str(floor(noiseSigma))])
imwrite(lowlight(:, :, show_band), 'sudocolor_lowlight.png')
hold on;
drawRectangleImage = rectangle('Position',[uperleft_x,uperleft_y,width,height],'LineWidth',4,'EdgeColor','r');
hold off;
title('Lowlight Image');
subplot(133), imshow(im_output(:, :, show_band));
%title(['MPSNR: ',num2str(mean(PSNR),'%2.4f'),'dB','    MSSIM: ',num2str(mean(SSIM),'%2.4f'),'    MSA: ',num2str(SAM1,'%2.4f')])
imwrite(im_output(:, :, show_band), 'sudocolor_retinexnet_refine_enhanced.png')
title('Enhanced Image')
drawnow;

disp([mean(PSNR), mean(SSIM), SAM1]);
