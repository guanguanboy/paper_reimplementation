%load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\lowlight_origin_outdoor_standard\test\1ms\007_2_2021-01-19_050.mat');
%load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\lowlight_origin_outdoor_standard\test\1ms\007_2_2021-01-20_015.mat');
%load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\lowlight_origin_outdoor_standard\test\1ms\007_2_2021-01-20_018.mat');
load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\lowlight_origin_outdoor_standard\test\1ms\007_2_2021-01-20_024.mat');

single_band_index = 30;
figure(1);
title('Low Light Image');
imshow(lowlight_normalized_hsi(:,:,single_band_index));
lowlight = lowlight_normalized_hsi;

%load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\lowlight_origin_outdoor_standard\test\15ms\007_2_2021-01-19_050.mat');
%load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\lowlight_origin_outdoor_standard\test\15ms\007_2_2021-01-20_015.mat');
%load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\lowlight_origin_outdoor_standard\test\15ms\007_2_2021-01-20_018.mat');
load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\lowlight_origin_outdoor_standard\test\15ms\007_2_2021-01-20_024.mat');

figure(2);
imshow(label_normalized_hsi(:,:,single_band_index));
im_label = label_normalized_hsi;

show_band=[57, 27, 17];

restored_57 = histeq(lowlight(:,:,57));
restored_27 = histeq(lowlight(:,:,27));
restored_17 = histeq(lowlight(:,:,17));

restored = cat(3,restored_57,restored_27,restored_17);
imshow(restored);
imwrite(restored, 'sudocolor_histeq_outdoor_024.png')


[w,h, band_num] = size(im_label);
restored_hsi = zeros(w, h, band_num);
for i=1:band_num
    restored_hsi(:,:,i) = histeq(lowlight(:,:,i));
    
end

im_output = restored_hsi;

%% PSNR & SSIM
PSNR=zeros(band_num, 1);
SSIM=zeros(band_num, 1);

for i=1:band_num
 
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
imwrite(im_label(:, :, show_band), 'sudocolor_label_outdoor_024.png')
title('Label Image')

subplot(132), imshow(lowlight(:, :, show_band));
%title(['Noise Level = ', num2str(floor(noiseSigma))])
imwrite(lowlight(:, :, show_band), 'sudocolor_lowlight_outdoor_024.png')
title('Lowlight Image');
subplot(133), imshow(im_output(:, :, show_band));
%title(['MPSNR: ',num2str(mean(PSNR),'%2.4f'),'dB','    MSSIM: ',num2str(mean(SSIM),'%2.4f'),'    MSA: ',num2str(SAM1,'%2.4f')])
imwrite(im_output(:, :, show_band), 'sudocolor_histeq_outdoor_024enhanced.png')
title('Enhanced Image')
drawnow;

denoised = restored_hsi;
save('testresult/HISTEQ_outdoor_024_enhanced.mat', 'denoised');

disp([mean(PSNR), mean(SSIM), SAM1]);
