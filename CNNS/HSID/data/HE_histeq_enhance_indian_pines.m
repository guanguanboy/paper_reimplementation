load('.\origin\IndianPines_Data_normalized_result.mat');

[w,h, band_num] = size(denoised);
restored_hsi = zeros(w, h, band_num);
lowlight = denoised;
for i=1:band_num
    restored_hsi(:,:,i) = histeq(lowlight(:,:,i));
    
end


denoised = restored_hsi;
save('testresult/indian/HE_indian_pines_indoor_enhanced.mat', 'denoised');