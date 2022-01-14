%噪声数据使用高斯加性噪声来模拟
%生成噪声数据并且保存起来

function save_patch(rotated_patch, savePath)
global count

%保存label数据
count = count + 1;
count_name = num2str(count, '%05d');

%加噪并保存noise数据
%patch_noise = imnoise(rotated_patch, 'gaussian')
%% noise level
noiseSigma=100.0;
[w,h, band] = size(rotated_patch);
for i=1:band
    patch_noise(:, :, i) = rotated_patch(:, :, i) + noiseSigma/255.0*randn(size(rotated_patch(:, :, i))); 
end

%保存patch_noise
label = rotated_patch;
noisy = patch_noise;
save([savePath,count_name,'.mat'], 'noisy', 'label');