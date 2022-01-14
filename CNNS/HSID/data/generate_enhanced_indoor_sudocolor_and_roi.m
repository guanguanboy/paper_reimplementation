
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
%load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\indoor_mat\lhsie_indoorlhsie_indoor_result.mat');
load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\indoor_mat\RetinexNet_indoor_lowlight_enhanced.mat');

method_name = 'RetinexNet';
%单通道图像及ROI生成
figure(3);
imshow(denoised(:,:,50));
title('EnLightened Image');
roi_path = 'D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\indoor_roi\';
%imwrite(denoised(:,:,50), 'lhsie_indoor_denoised.png')

band_50 = denoised(:,:,50);
imwrite(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y), [roi_path, method_name, '_enlarged_roi_indoor.png'])

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

%伪彩色图像生成
show_band=[57, 27, 17];
%show_band=[30, 20, 10];
figure(6);
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

drawnow;

rgb_path = 'D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\indoor_rgb\';

imwrite(im_output(:, :, show_band), [rgb_path, method_name, '_sudocolor_indoor_enhanced.png']);
figure(7);
imshow(im_output(:, :, show_band));
title('Enhanced RGB Image')

figure(8);
imshow(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y));
title('ROI');

disp([mean(PSNR), mean(SSIM), SAM1]);
