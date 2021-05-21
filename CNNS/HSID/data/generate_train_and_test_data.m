%wdc = imread('..\origin\dc.tif');
%save('../WDC/Hyperspectral_Project/wdc.mat','wdc');
load('./origin/wdc.mat')

%The gray values of each HSI band were all normalized[1].to [0,
%测试数据集的生成
%The first data set was the Washington dc Mall
%image mentioned above in Section IV-B, which
%was cropped to 200 × 200 for the simulated-data
%experiments.
maxvalue = double(max(max(max(wdc)))); %值为32720
minvalue = double(min(min(min(wdc)))); %值为-32728

%整理数据
MAX_VALUE = 4095
wdc(wdc>MAX_VALUE) = MAX_VALUE;%将数据中大于511的值全部替换为511
wdc(wdc<0) = 0; %将数据中小于0的值全部替换为0

    
double_wdc = double(wdc);
normalized_wdc = double_wdc./MAX_VALUE;
normalized_wdc_band150 = normalized_wdc(:,:,150);
figure(1);imshow(normalized_wdc(1:600,:, 150));
figure(2);imshow(normalized_wdc(601:1280,:, 150));
part_two = normalized_wdc(601:1280,:,:);
crop_test  = part_two(1:200,51:250,:); %crop_test基本与GT_crop一致，裁剪坐标为1:200,51:250
disp('mean of crop_test: ')
mean_crop_test = mean(mean(mean(crop_test))) %当MAXVALUE为4095时，该值与GT_crop的值接近0.4392
%当MAXVALUE为511时，crop_test150的显示效果与GT_crop.mat 150band的显示效果最接近
load('./test/GT_crop.mat');
mean_GT_crop = mean(mean(mean(temp)))%0.4210
crop_test150 = crop_test(:,:,150); 
figure(3);imshow(crop_test(:,:,150));

%训练数据集的生成
savePath = './train/';

global count
count = 0;
%大小，1280*307*191
wdc_training_data = [normalized_wdc(1:600,:,:);normalized_wdc(801:1280,:,:)];
% These training
% data were then cropped in each patch size as 20×20,
% with the stride equal to 20. The simulated noisy patches
% are generated through imposing additive white Gaussian
% noise with different spectrums. The noise intensity is
% multiple and conforms to a fixed distribution or random
% probability distribution for different experiments. From
% the point of view of increasing the number of HSI
% training samples to better fit the HSI denoising mode,
%要点：
% multiangle image rotation (angles of 0°, 90°, 180°,
% and 270°) and multiscale resizing (scales of 0.5, 1, 1.5,
% and 2 in our training data sets) were both utilized during
% the training procedure.
%总结，将1280*307 切分为20*20的patch，stride取20
%旋转4个角度，使用4个scale。
%噪声数据使用高斯加性噪声来模拟
scales = [0.5, 1, 1.5, 2];
for sc = 1:length(scales) %遍历四个scale
    scaled_wdc_training_data = imresize(wdc_training_data, scales(sc));
    size(scaled_wdc_training_data)
    width = size(scaled_wdc_training_data, 1);
    width_divide_20 = width - mod(width,20); %这一步操作是为了将最后一个小于20*20的patch去掉
    height = size(scaled_wdc_training_data, 2);
    height_divide_20 = height - mod(height,20);%这一步操作是为了将最后一个小于20*20的patch去掉
    x_index = 1:20:width_divide_20;
    y_index = 1:20:height_divide_20;
    
    for i = 1:length(x_index)
        for j = 1:length(y_index)
            patch = scaled_wdc_training_data(x_index(i):x_index(i)+20-1,y_index(j):y_index(j)+20-1,:);
            rotate_patch(patch, savePath);
        end
    end
    
end



%display generated nosiy and label img
%figure(4);imshow(label(:,:,50))
%figure(5);imshow(noisy(:,:,50))
