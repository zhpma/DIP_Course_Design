function patch = Cutting( src, patch_size )
% CUTTING 从源图像中提取指定大小的图像块
%   输入参数：
%   - src: 输入图像
%   - patch_size: 图像块的尺寸
%
%   输出参数：
%   - patch: 提取的图像块

    patch_size_half = (patch_size - 1) / 2;  % 计算图像块尺寸的一半
    
    [h, w, d] = size(src);  % 获取输入图像的尺寸
    
    if patch_size == 21
        if d == 3
            src = rgb2ycbcr(src);  % 如果输入图像为彩色图像，将其转换为YCbCr颜色空间
            src = src(:, :, 1);  % 仅保留亮度通道
        end
        nh = h - mod(h, 3);  % 计算高度调整后的尺寸
        nw = w - mod(w, 3);  % 计算宽度调整后的尺寸
        new_src = src(1 : nh, 1 : nw);  % 调整后的图像
        [h, w] = size(new_src);  % 更新图像尺寸
    end
    
    if patch_size == 21
        num = ((h - patch_size) / 3 + 1) * ((w - patch_size) / 3 + 1);  % 计算图像块数量
    else
        num = (h - patch_size + 1) * (w - patch_size + 1);  % 计算图像块数量
    end
    
    patch = zeros(num, patch_size * patch_size);  % 初始化图像块矩阵
    
    flag = 1;  % 初始化标记变量
    
    if patch_size == 7
        for i = patch_size_half + 1 : h - patch_size_half
            for j = patch_size_half + 1 : w - patch_size_half
                temp_patch = src(i - patch_size_half : i + patch_size_half, j - patch_size_half : j + patch_size_half);  % 提取图像块
                temp_patch = reshape(temp_patch, 1, patch_size * patch_size);  % 将图像块重塑为一维向量
                patch(flag, :) = temp_patch;  % 将图像块存储到输出矩阵中
                flag = flag + 1;  % 更新标记变量
            end
        end
    else
        for i = patch_size_half + 1 : 3 : h - patch_size_half
            for j = patch_size_half + 1 : 3 : w - patch_size_half
                temp_patch = src(i - patch_size_half : i + patch_size_half, j - patch_size_half : j + patch_size_half);  % 提取图像块
                temp_patch = reshape(temp_patch, 1, patch_size * patch_size);  % 将图像块重塑为一维向量
                patch(flag, :) = temp_patch;  % 将图像块存储到输出矩阵中
                flag = flag + 1;  % 更新标记变量
            end
        end
    end
    
end
