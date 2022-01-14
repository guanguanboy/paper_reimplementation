
%加噪level 25的noise数据
load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\test\GT_crop_noise_level25.mat');
figure(1);
imshow(noisy(:,:,50));

figure(4);
imshow(label(:,:,50));
%加噪去噪后的数据
load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\result34.mat');
%load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\result35_hsidcnn.mat');
figure(2);
imshow(denoised(:,:,50));

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
figure(3);
subplot(131), imshow(im_label(:, :, show_band));
title(['Original Band Number: ', num2str(show_band)])

%subplot(132), imshow(im_input(:, :, show_band));
%title(['Noise Level = ', num2str(floor(noiseSigma))])

subplot(133), imshow(im_output(:, :, show_band));
title(['MPSNR: ',num2str(mean(PSNR),'%2.4f'),'dB','    MSSIM: ',num2str(mean(SSIM),'%2.4f'),'    MSA: ',num2str(SAM1,'%2.4f')])

drawnow;

disp([mean(PSNR), mean(SSIM), SAM1]);
