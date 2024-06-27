function hr = Generate_HR(lr)
% Generate_HR 生成高分辨率图像
% 该函数从低分辨率图像中生成高分辨率图像。

    % 在低分辨率图像的周围添加边界，以便进行图像块切割
    lr = padarray(lr, [2, 2], 'replicate', 'both');
    
    % 切割低分辨率图像块
    lr_patches = Cutting(lr, 7);
    
    % 提取低分辨率图像块的特征
    lr_features = Get_Feature(lr_patches);
    
    % 加载预训练的系数矩阵
    load('../lib/coef.mat');
    
    % 转置每个系数矩阵
    for i = 1 : 512
        coef(:, :, i) = Coef(:, :, i)';
    end
    
    % 加载预训练的中心点
    load('../lib/center.mat');
    
    [patch_num, patch_size_pow] = size(lr_patches); % 获取图像块数量和尺寸
    [lr_height, lr_width] = size(lr); % 获取低分辨率图像的高度和宽度
    
    % 计算高分辨率图像的尺寸
    h_num = lr_height - 7 + 1;
    w_num = lr_width - 7 + 1;
    
    hr = zeros((lr_height - 4) * 3, (lr_width - 4) * 3); % 初始化高分辨率图像矩阵
    count = zeros((lr_height - 4) * 3, (lr_width - 4) * 3); % 初始化计数矩阵
    
    % 遍历每个低分辨率图像块
    for i = 1 : patch_num
        lr_patch = lr_features(i, :); % 获取当前图像块特征
        temp = repmat(lr_patch, 512, 1);
        diff = temp - C; % 计算与中心点的差距
        diff = diff.^2;
        distance = sum(diff, 2); % 计算欧氏距离
        [~, id] = min(distance); % 找到最近的中心点
        hr_patch = lr_patch * coef(:, :, id); % 使用对应的系数矩阵生成高分辨率图像块
        
        % 去除低分辨率图像块的边界，并计算均值
        lr_cut = [2:6 8:42 44:48];
        lr_temp = lr_patches(i, :);
        lr_temp = lr_temp(lr_cut);
        lr_temp = double(lr_temp);
        lr_mean = mean(lr_temp);
        
        % 调整高分辨率图像块的均值
        hr_patch = reshape(hr_patch, 9, 9) + lr_mean;
        
        % 计算当前图像块在高分辨率图像中的位置
        r = ceil(i / w_num);
        c = mod(i - 1, w_num) + 1;
        
        rh = (r - 1) * 3 + 1;
        rh1 = rh + 9 - 1;
        ch = (c - 1) * 3 + 1;
        ch1 = ch + 9 - 1;
        
        % 将高分辨率图像块加到高分辨率图像中
        hr(rh : rh1, ch : ch1) = hr(rh : rh1, ch : ch1) + hr_patch;
        count(rh : rh1, ch : ch1) = count(rh : rh1, ch : ch1) + 1; % 更新计数
    end
    
    % 计算高分辨率图像的平均值
    hr = hr ./ count;
    hr = uint8(hr); % 转换为8位无符号整数类型

end
