load('D:\DataSets\hyperspectraldatasets\lowlight_hyperspectral_datasets\lowlight\test\soup_bigcorn_orange_1ms.mat');

uperleft_x = 131;
uperleft_y = 141;
lowerright_x = 250;
lowerright_y = 260;
width = 120;
height = 120;

pt = [131, 141];
wSize = [120,120];

show_band=[57, 27, 17];
%show_band=[30, 20, 10];
figure(4);
lowlight_rgb = lowlight(:, :, show_band);
lowlight_roi = drawRect(lowlight_rgb, pt,wSize,5, [255, 0, 0]);
imshow(lowlight_roi);
%title(['Noise Level = ', num2str(floor(noiseSigma))])
imwrite(lowlight_roi, 'lowlight_origin_roi.png');
title('Lowlight Image');