function [fft_img] = fft_for_show(gray_img)
I=im2double(gray_img);%将矩阵转化为double型,图像计算中很多处理不能用整型
F=fft2(I);%进行二维傅里叶变换
F=fftshift(F); %fftshift将傅里叶变换的零频率成分移到频谱中心，因为fft2变换中，信号的零频率成分在信号左上角。
F=abs(F);
fft_img=log(F+1); %加对数以便于显示图像