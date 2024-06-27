function feature = Get_Feature(patch)
% Get_Feature 提取图像块的特征
% 输入:
%    patch - 图像块的矩阵
% 输出:
%    feature - 提取的特征矩阵

    % 获取图像块的数量和每个图像块的尺寸
    [num, pow_ps] = size(patch);
    
    % 计算图像块的边长
    patch_size = sqrt(pow_ps);
    
    % 如果图像块的边长为7
    if patch_size == 7
        % 提取用于计算特征的像素索引
        lr_cut = [2:6 8:42 44:48];
        feature = zeros(num, 45); % 初始化特征矩阵
        
        for i = 1 : num
            % 获取单个图像块并重塑为7x7
            lr_patch = patch(i, :);
            lr_patch = reshape(lr_patch, 7, 7);
            
            % 提取特定索引的像素值
            new_lr_patch = lr_patch(lr_cut);
            new_lr_patch = double(new_lr_patch); % 转换为双精度类型
            
            % 计算图像块的均值并去均值
            patch_mean = mean2(new_lr_patch);
            feature(i, :) = reshape(new_lr_patch - patch_mean, 1, 45); % 将去均值后的图像块重塑为一维向量
        end
    else
        % 如果图像块的边长不是7（默认为21）
        feature = zeros(num, 81); % 初始化特征矩阵
        
        for i = 1 : num
            % 获取单个图像块并重塑为21x21
            hr_patch = patch(i, :);
            hr_patch = reshape(hr_patch, 21, 21);
            
            % 提取图像块的中间区域7:15, 7:15
            new_hr_patch = hr_patch(7:15, 7:15);
            new_hr_patch = double(new_hr_patch); % 转换为双精度类型
            
            % 将提取的中间区域重塑为一维向量
            feature(i, :) = reshape(new_hr_patch, 1, 81);
        end
    end
end
