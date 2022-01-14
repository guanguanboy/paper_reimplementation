lowlight_img = imread('lowlight.png');
lowlight_edge = edge(lowlight_img, 'log', 0.003, 2);
figure(1);
subplot(121);
imshow(lowlight_img);
title('lowlight image');

histeq = histeq(lowlight_img);
figure(1);
subplot(122);
imshow(histeq);
title('histeq image')

% add_img = lowlight_img + 100;
% figure(1);
% subplot(133);
% imshow(add_img);
% title('add image');
