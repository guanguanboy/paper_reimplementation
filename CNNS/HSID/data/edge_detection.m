lowlight_img = imread('lowlight.png');
lowlight_edge = edge(lowlight_img, 'log', 0.003, 2);
figure(1);
imshow(lowlight_edge);
title('lowlight edge');

denoised_img = imread('denoised.png');
denoised_edge = edge(denoised_img, 'canny', 0.003, 2);
figure(2);
imshow(denoised_edge, []);
title('denoised');

label_img=imread('label.png');
label_edge = edge(label_img, 'canny', 0.003, 2);
%label_edge = edge(label_img(100:120,100:120), 'log', 0.003, 2);

figure(3);
imshow(label_edge,[]);
title('label');
