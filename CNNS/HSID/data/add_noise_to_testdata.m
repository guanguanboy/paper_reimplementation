load('./origin/GT_crop.mat');
im_label=temp;
[w,h, band] = size(im_label);
im_input=zeros(w,h, band);

%% noise level
noiseSigma=5.0;

%% add noise (same level for all bands)
%% 方式是将一个randn,randn中的数据满足均值为0， 标准差为1.的矩阵，乘以noiseSigma，除以255.0，然后再加上原来矩阵的值，就得到了加噪后的矩阵
for i=1:band
 
    im_input(:, :, i) = im_label(:, :, i) + noiseSigma/255.0*randn(size(im_label(:, :, i))); 
end

noisy = im_input;
label = temp;
%%将im_input保存成mat格式
savePath = './test/';
test_noise_name = 'GT_crop_noise_level_5';
save([savePath,test_noise_name,'.mat'], 'noisy', 'label');
imshow(noisy(:,:,150))
