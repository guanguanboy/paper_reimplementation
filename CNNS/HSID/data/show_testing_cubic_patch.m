load('D:\Codes\codes_of_paper_reimplementation\CNNS\HSID\data\190.mat');
figure(1);
imshow(noisy);
figure(2);
imshow(label);
figure(5);
cubic_band5 = cubic(6,:,:);
cubic_band5 = permute(cubic_band5, [2, 3, 1]);
imshow(cubic_band5);