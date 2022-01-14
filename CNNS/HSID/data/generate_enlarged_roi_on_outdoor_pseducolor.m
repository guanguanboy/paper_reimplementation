uperleft_x = 11;
uperleft_y = 251;
lowerright_x = 170;
lowerright_y = 410;

%加噪去噪后的数据
load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\hsid_origin.mat');
band_50 = denoised(:,:,50);
imwrite(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y), 'enlarged_roi_hsid_ablation_l2loss.png');

load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\hsid_rdn_eca_l1_loss_600epoch_patchsize32_best_10_23_k24');
band_50 = denoised(:,:,50);
imwrite(band_50(131:250, 141:260), 'enlarged_roi_rdecab_ablation_l1loss.png');


load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\hsid_rdecab_ablation_l2loss.mat');
band_50 = denoised(:,:,50);
imwrite(band_50(131:250, 141:260), 'enlarged_roi_rdecab_ablation_l2loss.png');

load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\result_encam.mat');
band_50 = denoised(:,:,50);
imwrite(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y), 'enlarged_roi_encam.png');

load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\HISTEQ_enhanced.mat');
band_50 = denoised(:,:,50);
imwrite(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y), 'enlarged_roi_HISTEQ.png');

load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\BM4D_enhanced.mat');
band_50 = denoised(:,:,50);
imwrite(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y), 'enlarged_roi_BM4D.png');

load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\LRMR_enhanced.mat');
band_50 = denoised(:,:,50);
imwrite(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y), 'enlarged_roi_LRMR.png');

load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\LRTA_enhanced.mat');
band_50 = denoised(:,:,50);
imwrite(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y), 'enlarged_roi_LRTA.png');

load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\testresult\LRTV_enhanced.mat');
band_50 = denoised(:,:,50);
imwrite(band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y), 'enlarged_roi_LRTV.png');

load('D:\DataSets\hyperspectraldatasets\lowlight_hyperspectral_datasets\lowlight\test\soup_bigcorn_orange_1ms.mat');
lowlight_band_50 = lowlight(:,:,50);
label_band_50 = label(:,:,50);
imwrite(lowlight_band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y), 'enlarged_roi_lowlight.png');
imwrite(label_band_50(uperleft_x:lowerright_x, uperleft_y:lowerright_y), 'enlarged_roi_label.png');

figure(3);
title('EnLightened Image');
imshow(band_50);