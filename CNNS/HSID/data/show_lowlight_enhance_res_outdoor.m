
load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\lowlight_origin_outdoor_standard\test\1ms\007_2_2021-01-19_050.mat');

single_band_index = 30;
figure(1);
title('Low Light Image');
imshow(lowlight_normalized_hsi(:,:,single_band_index));

hold on;
uperleft_x = 131;
uperleft_y = 141;
lowerright_x = 250;
lowerright_y = 260;
width = 120;
height = 120;
drawRectangleImage = rectangle('Position',[uperleft_x,uperleft_y,width,height],'LineWidth',4,'EdgeColor','r');
hold off;
imwrite(lowlight_normalized_hsi(:,:,single_band_index), 'lowlight_outdoor_007_2_2021-01-19_050.png')

load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\lowlight_origin_outdoor_standard\test\15ms\007_2_2021-01-19_050.mat');
figure(2);
imshow(label_normalized_hsi(:,:,single_band_index));
title('Label Image');
imwrite(label_normalized_hsi(:,:,single_band_index), 'label_outdoor_007_2_2021-01-19_050.png');

%加噪去噪后的数据
load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\Ablationstudy_result\outdoor\lshie_outdoor_007_2_2021-01-19_050_best.mat');
%load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\hsid_rdn_eca_l1_loss_600epoch_patchsize32_best_10_23_k24');
%load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\hsid_rdecab_ablation_l2loss.mat');


figure(3);
imshow(denoised(:,:,single_band_index));
title('EnLightened Image');
imwrite(denoised(:,:,single_band_index), 'denoised.png')
band_50 = denoised(:,:,single_band_index);
imwrite(band_50(131:250, 141:260), 'enlarged_roi_hsid_ablation_l2loss.png')
%imwrite(band_50(131:250, 141:260), 'enlarged_roi_rdecab_ablation_l1loss.png')
%imwrite(band_50(131:250, 141:260), 'enlarged_roi_rdecab_ablation_l2loss.png')

figure(5);
title('residual');
imshow(label_normalized_hsi(:,:,single_band_index) - denoised(:,:,single_band_index))
im_label = label_normalized_hsi;
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

%show_band=[57, 27, 17];
show_band=[37, 23, 10];
figure(4);
subplot(131), imshow(im_label(:, :, show_band));
imwrite(im_label(:, :, show_band), 'sudocolor_label_007_2_2021-01-19_050.png')
title('Label Image')

subplot(132), imshow(lowlight_normalized_hsi(:, :, show_band));
%title(['Noise Level = ', num2str(floor(noiseSigma))])
imwrite(lowlight_normalized_hsi(:, :, show_band), 'sudocolor_lowlight_007_2_2021-01-19_050.png')
hold on;
drawRectangleImage = rectangle('Position',[uperleft_x,uperleft_y,width,height],'LineWidth',4,'EdgeColor','r');
hold off;
title('Lowlight Image');
subplot(133), imshow(im_output(:, :, show_band));
%title(['MPSNR: ',num2str(mean(PSNR),'%2.4f'),'dB','    MSSIM: ',num2str(mean(SSIM),'%2.4f'),'    MSA: ',num2str(SAM1,'%2.4f')])
imwrite(im_output(:, :, show_band), 'sudocolor_lshie_outdoor_007_2_2021-01-19_050_enhanced.png')
title('Enhanced Image')
drawnow;

disp([mean(PSNR), mean(SSIM), SAM1]);
