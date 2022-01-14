ROI_list = [131, 250, 141, 260];
ROI_list_int = round(ROI_list);
uperleft_x = 250;
uperleft_y = 11;
lowerright_x = 410;
lowerright_y = 170;

root_path_read = 'testresult/outdoor_result/pseudo_color';
root_path_save = 'testresult/outdoor_result/pseudo_color_roi';

filelist=dir(root_path_read);%get the filelist from rootpath
[filenum,temp]=size(filelist);%get the filelist's count

for i=1:filenum
    if strcmp(filelist(i).name,'.')|| strcmp(filelist(i).name,'..')
    %do nothing
    else
        read_filename = strcat(root_path_read, '/', filelist(i).name);
        im_pseudocolor_output = imread(read_filename);
        save_filename = strcat(root_path_save, '/', 'roi_', filelist(i).name);
        imwrite(im_pseudocolor_output(uperleft_x:lowerright_x, uperleft_y:lowerright_y,:), save_filename);
    end
end

%im_pseudocolor_output = imread('testresult/outdoor_result/pseudo_color/LRMR_007_2_2021-01-20_018.png');
%imwrite(im_pseudocolor_output(uperleft_x:lowerright_x, uperleft_y:lowerright_y,:), 'testresult/outdoor_result/pseudo_color_roi/roi_LRMR_007_2_2021-01-20_018.png');
pt = [11, 251];
wSize = [160,160];

rgb_img = imread('testresult/outdoor_result/pseudo_color/sudocolor_lowlight_007_2_2021-01-20_018.png');
lowlight_roi = drawRect(rgb_img, pt,wSize,5, [255, 0, 0]);
imshow(lowlight_roi);
%title(['Noise Level = ', num2str(floor(noiseSigma))])
imwrite(lowlight_roi, 'testresult/outdoor_result/pseudo_color_roi/outdoor_origin_lowlight_roi.png');
