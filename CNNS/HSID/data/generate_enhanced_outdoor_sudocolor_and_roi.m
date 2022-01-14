rng('default');

%预设参数
show_band=[37, 23, 10];
%show_band=[57, 27, 17];

uperleft_x = 193;
uperleft_y = 129;
lowerright_x = 448;
lowerright_y = 384;
width = 256;
height = 256;
rgb_path = 'D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\outdoor_result_rgb\';
roi_path = 'D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\outdoor_result_roi\';

%加载低光照的数据
file_name = '007_2_2021-01-20_024';
file_type = '.mat';
file_prefix = 'D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\lowlight_origin_outdoor_standard\test\1ms\';
file_path = sprintf('%s%s%s', file_prefix, file_name, file_type);
load(file_path);

lowlight_normalized_hsi = rot90(lowlight_normalized_hsi,3);
figure(1);
imshow(lowlight_normalized_hsi(:,:,show_band));
title('lowlight origin rgb')
hold on;
drawRectangleImage = rectangle('Position',[uperleft_x,uperleft_y,width,height],'LineWidth',4,'EdgeColor','r');
hold off;

%imwrite(lowlight_normalized_hsi(:,:,show_band), [rgb_path, 'lowlight_outdoor_024.png']);
%band_50 = lowlight_normalized_hsi(:,:,50);
%imwrite(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y), [roi_path, 'lowlight_024_enlarged_roi_outdoor.png'])
%figure(9);
%imshow(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y));



%加载Label数据
file_prefix = 'D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\lowlight_origin_outdoor_standard\test\15ms\';
file_path = sprintf('%s%s%s', file_prefix, file_name, file_type);
load(file_path);
label_normalized_hsi = rot90(label_normalized_hsi,3);

figure(2);
imshow(label_normalized_hsi(:,:,show_band));
title('label rgb')

%imwrite(label_normalized_hsi(:,:,show_band), [rgb_path, 'label_outdoor_024.png']);
%band_50 = label_normalized_hsi(:,:,50);
%imwrite(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y), [roi_path, 'label_024_enlarged_roi_outdoor.png'])
%figure(10);
%imshow(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y));

%加载增强后的数据
file_name = 'ENCAM_outdoor_024_enhanced_upsampled.mat';
file_prefix = 'D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\outdoor_result_mat\';
file_path = sprintf('%s%s%s', file_prefix, file_name);
load(file_path);
denoised = rot90(denoised, 3);

im_denoised = denoised;
figure(3);
imshow(im_denoised(:, :, show_band));

method_name = 'MSR';
%单通道图像及ROI生成
figure(4);
imshow(denoised(:,:,50));
title('EnLightened Image');

band_50 = denoised(:,:,50);
imwrite(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y), [roi_path, method_name, '_enlarged_roi_outdoor.png'])

figure(5);
title('residual');
label = label_normalized_hsi;
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
figure(6);
subplot(131), imshow(im_label(:, :, show_band));
%imwrite(im_label(:, :, show_band), 'sudocolor_label.png')
title('Label Image')

lowlight = lowlight_normalized_hsi;
subplot(132), imshow(lowlight(:, :, show_band));
%title(['Noise Level = ', num2str(floor(noiseSigma))])
%imwrite(lowlight(:, :, show_band), 'sudocolor_lowlight.png')
hold on;
drawRectangleImage = rectangle('Position',[uperleft_x,uperleft_y,width,height],'LineWidth',4,'EdgeColor','r');
hold off;
title('Lowlight Image');
subplot(133), imshow(im_output(:, :, show_band));

drawnow;


imwrite(im_output(:, :, show_band), [rgb_path, method_name, '_sudocolor_outdoor_enhanced.png']);
figure(7);
imshow(im_output(:, :, show_band));
title('Enhanced RGB Image')

%label_normalized_hsi = rot90(label_normalized_hsi, 3);
show_band_output = lowlight_normalized_hsi(:, :, show_band);
figure(8);
imshow(show_band_output(uperleft_x:lowerright_x, uperleft_y:lowerright_y,:));
title('ROI');

disp([mean(PSNR), mean(SSIM), SAM1]);
