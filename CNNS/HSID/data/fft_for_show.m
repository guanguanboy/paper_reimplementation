function [fft_img] = fft_for_show(gray_img)
I=im2double(gray_img);%������ת��Ϊdouble��,ͼ������кܶദ����������
F=fft2(I);%���ж�ά����Ҷ�任
F=fftshift(F); %fftshift������Ҷ�任����Ƶ�ʳɷ��Ƶ�Ƶ�����ģ���Ϊfft2�任�У��źŵ���Ƶ�ʳɷ����ź����Ͻǡ�
F=abs(F);
fft_img=log(F+1); %�Ӷ����Ա�����ʾͼ��