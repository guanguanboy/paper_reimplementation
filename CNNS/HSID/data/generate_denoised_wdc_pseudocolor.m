load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\Ablationstudy_result\wdc_denoised_result_level25\washington_ENCAM.mat');
uperleft_x = 51;
uperleft_y = 81;
lowerright_x = 130;
lowerright_y = 160;
show_band=[57, 27, 17];

figure(1);
washingtons_pseudocolor = washingtons(:, :, show_band);
imshow(washingtons_pseudocolor);
imwrite(washingtons_pseudocolor, 'testresult/Ablationstudy_result/wdc_denoised_result_level25/images_png_format/wdc_encam_denoise.png')
title(['Original Band Number: ', num2str(show_band)])
imwrite(washingtons_pseudocolor(uperleft_x:lowerright_x, uperleft_y:lowerright_y,:), 'testresult/Ablationstudy_result/wdc_denoised_result_level25/enlarged_roi/enlarged_roi_wdc_encam_denoise.png')


load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\Ablationstudy_result\wdc_denoised_result_level25\wdc_hsid.mat');
denoised_pseudocolor = denoised(:, :, show_band);
imwrite(denoised_pseudocolor, 'testresult/Ablationstudy_result/wdc_denoised_result_level25/images_png_format/wdc_hsid_denoise.png')
figure(2);
imshow(denoised_pseudocolor);
imwrite(denoised_pseudocolor(uperleft_x:lowerright_x, uperleft_y:lowerright_y,:), 'testresult/Ablationstudy_result/wdc_denoised_result_level25/enlarged_roi/enlarged_roi_wdc_hsid_denoise.png')

load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\Ablationstudy_result\wdc_denoised_result_level25\LHSIE_denoise_wdc_level25.mat');
denoised_pseudocolor = denoised(:, :, show_band);
imwrite(denoised_pseudocolor, 'testresult/Ablationstudy_result/wdc_denoised_result_level25/images_png_format/wdc_LSHIE_denoise.png')
figure(3);
imshow(denoised_pseudocolor);
imwrite(denoised_pseudocolor(uperleft_x:lowerright_x, uperleft_y:lowerright_y,:), 'testresult/Ablationstudy_result/wdc_denoised_result_level25/enlarged_roi/enlarged_roi_wdc_LSHIE_denoise.png')

load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\Ablationstudy_result\wdc_denoised_result_level25\GT_crop_noise25.mat');
origin_noisy_pseudocolor = noisy(:, :, show_band);
imwrite(origin_noisy_pseudocolor, 'testresult/Ablationstudy_result/wdc_denoised_result_level25/images_png_format/wdc_origin_noisy_level25.png')
imwrite(origin_noisy_pseudocolor(uperleft_x:lowerright_x, uperleft_y:lowerright_y,:), 'testresult/Ablationstudy_result/wdc_denoised_result_level25/enlarged_roi/enlarged_roi_wdc_origin_noisy_level25.png')
figure(4);
imshow(origin_noisy_pseudocolor);

pt = [51, 81];
wSize = [80,80];

lowlight_roi = drawRect(origin_noisy_pseudocolor, pt,wSize,5, [255, 0, 0]);
imshow(lowlight_roi);
%title(['Noise Level = ', num2str(floor(noiseSigma))])
imwrite(lowlight_roi, 'wdc_origin_noisy_roi.png');

drawnow;