%function Get_Coef_Matrix()
% Get_Coef_Matrix 生成系数矩阵
% 该函数从训练图像中提取低分辨率和高分辨率图像块，并计算它们的特征，
% 然后通过聚类算法计算每个类的系数矩阵并保存。

tic; % 开始计时
lr_sample = zeros(2000000, 7*7); % 初始化低分辨率样本矩阵
hr_sample = zeros(2000000, 21*21); % 初始化高分辨率样本矩阵
file_path = '../Train/'; % 训练图像路径
img_path_list = dir(strcat(file_path, '*.jpg')); % 获取所有训练图像文件列表
img_num = length(img_path_list); % 训练图像数量
sign = 1; % 样本计数器

% 遍历所有训练图像
for i = 1 : img_num
    image_name = img_path_list(i).name; % 获取图像文件名
    image = imread(strcat(file_path, image_name)); % 读取图像
    image = double(image); % 转换为双精度类型
    lr = HR_To_LR(image, 1.2); % 生成低分辨率图像
    lr_patches = Cutting(lr, 7); % 切割低分辨率图像块
    hr_patches = Cutting(image, 21); % 切割高分辨率图像块

    [patch_num, size_pow] = size(lr_patches); % 获取图像块数量和尺寸

    % 如果样本数量超出预设数量，截取前2000000个样本
    if patch_num + sign > 2000000
        lr_sample(sign : 2000000, :) = lr_patches(1 : 2000000 - sign + 1, :);
        hr_sample(sign : 2000000, :) = hr_patches(1 : 2000000 - sign + 1, :);
        break;
    else
        lr_sample(sign : sign + patch_num - 1, :) = lr_patches; % 保存低分辨率样本
        hr_sample(sign : sign + patch_num - 1, :) = hr_patches; % 保存高分辨率样本
        sign = sign + patch_num; % 更新样本计数器
    end 
end

lr_features = Get_Feature(lr_sample); % 提取低分辨率特征
hr_features = Get_Feature(hr_sample); % 提取高分辨率特征

% 减去低分辨率图像块的均值
for i = 1 : 2000000
    hr_old_features = hr_features(i, :);
    lr_temp = lr_sample(i, :);
    lr_temp = reshape(lr_temp, 7, 7);
    lr_cut = [2:6 8:42 44:48];
    lr_temp = lr_temp(lr_cut);
    lr_temp = double(lr_temp);
    lr_mean = mean2(lr_temp);
    hr_features(i, :) = hr_old_features - lr_mean;
end

load('../lib/center.mat'); % 加载预训练的中心点

idx = zeros(2000000, 1); % 初始化索引矩阵

% 计算每个低分辨率特征到中心点的距离，找到最近的中心点
for i = 1 : 2000000
    lr_patch = lr_features(i, :);
    temp = repmat(lr_patch, 512, 1);
    diff = temp - C;
    diff = diff.^2;
    distance = sum(diff, 2);
    [~, id] = min(distance);
    idx(i, 1) = id;
end

% 计算每个类的系数矩阵
for i = 1 : 512
    Coef(:, :, i) = hr_features(idx == i, :)' / lr_features(idx == i, :)';
end

save('../lib/coef.mat', 'Coef'); % 保存系数矩阵
toc; % 结束计时
%end
