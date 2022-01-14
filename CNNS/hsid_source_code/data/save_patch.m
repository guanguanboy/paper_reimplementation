%��������ʹ�ø�˹����������ģ��
%�����������ݲ��ұ�������

function save_patch(rotated_patch, savePath)
global count

%����label����
count = count + 1;
count_name = num2str(count, '%05d');

%���벢����noise����
%patch_noise = imnoise(rotated_patch, 'gaussian')
%% noise level
noiseSigma=100.0;
[w,h, band] = size(rotated_patch);
for i=1:band
    patch_noise(:, :, i) = rotated_patch(:, :, i) + noiseSigma/255.0*randn(size(rotated_patch(:, :, i))); 
end

%����patch_noise
label = rotated_patch;
noisy = patch_noise;
save([savePath,count_name,'.mat'], 'noisy', 'label');