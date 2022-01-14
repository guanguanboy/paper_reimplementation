lowlight_img = imread('lowlight.png');
lowlight_img_fft = fft_for_show(lowlight_img);
figure(1);
imshow(lowlight_img_fft);
title('lowlight');

denoised_img = imread('denoised.png');
dnoised_img_fft = fft_for_show(denoised_img);
figure(2);
imshow(dnoised_img_fft, []);
title('denoised');

I=imread('label.png');
%figure(1);
%imshow(I);
%I=rgb2gray(I);%将三维图像转化为二维
I=im2double(I);%将矩阵转化为double型,图像计算中很多处理不能用整型
F=fft2(I);%进行二维傅里叶变换
F=fftshift(F); %fftshift将傅里叶变换的零频率成分移到频谱中心，因为fft2变换中，信号的零频率成分在信号左上角。
F=abs(F);
T=log(F+1); %加对数以便于显示图像

figure(3);
imshow(T,[]);
title('label');
