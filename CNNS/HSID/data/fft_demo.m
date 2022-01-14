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
%I=rgb2gray(I);%����άͼ��ת��Ϊ��ά
I=im2double(I);%������ת��Ϊdouble��,ͼ������кܶദ����������
F=fft2(I);%���ж�ά����Ҷ�任
F=fftshift(F); %fftshift������Ҷ�任����Ƶ�ʳɷ��Ƶ�Ƶ�����ģ���Ϊfft2�任�У��źŵ���Ƶ�ʳɷ����ź����Ͻǡ�
F=abs(F);
T=log(F+1); %�Ӷ����Ա�����ʾͼ��

figure(3);
imshow(T,[]);
title('label');
