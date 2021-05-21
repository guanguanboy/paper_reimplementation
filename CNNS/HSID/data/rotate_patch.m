%multiangle image rotation (angles of 0°, 90°, 180°,
% and 270°)
function rotate_patch(original_patch, savePath)
angles = [0, 90, 180, 270]

for i=1:length(angles)
    rotated_patch = imrotate(original_patch, angles(i));
    save_patch(rotated_patch, savePath);
end

%不旋转
%旋转90度
%旋转180度
 %旋转270度