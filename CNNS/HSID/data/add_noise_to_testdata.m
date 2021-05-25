load('./origin/GT_crop.mat');
im_label=temp;
[w,h, band] = size(im_label);
im_input=zeros(w,h, band);

%% noise level
noiseSigma=25.0;

%% add noise (same level for all bands)
%% ��ʽ�ǽ�һ��randn�ľ��󣬳���noiseSigma������255.0��Ȼ���ټ���ԭ�������ֵ���͵õ��˼����ľ���
for i=1:band
 
    im_input(:, :, i) = im_label(:, :, i) + noiseSigma/255.0*randn(size(im_label(:, :, i))); 
end

noisy = im_input;
label = temp;
%%��im_input�����mat��ʽ
savePath = './test/';
test_noise_name = 'GT_crop_noise';
save([savePath,test_noise_name,'.mat'], 'noisy', 'label');
imshow(noisy(:,:,150))
