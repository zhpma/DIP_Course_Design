%function Train_Center()
% Train_Center 函数用于从训练图像中提取低分辨率图像块，并基于这些图像块进行特征提取和聚类
% 输入：
%    无输入参数
% 输出：
%    无显式输出，但会在运行过程中显示处理时间，并进行特征提取和聚类

tic; % 开始计时

% 初始化存储样本的矩阵，200000个样本，每个样本是7x7的图像块
sample = zeros(200000, 7*7); 

% 设置图像文件路径
file_path = 'E:\DIP\综合\Super_Resolution-master2\Train'; 

% 获取目录下所有jpg图像文件的列表
img_path_list = dir(strcat(file_path, '*.jpg')); 

% 获取图像数量
img_num = length(img_path_list); 

% 初始化计数器，用于记录已处理的图像块数量
sign = 1; 

% 遍历所有图像
for i = 1 : img_num
    % 获取图像文件名
    image_name = img_path_list(i).name; 

    % 读取图像
    image = imread(strcat(file_path, image_name)); 

    % 将高分辨率图像转换为低分辨率图像，缩放比例为1.2
    lr = HR_To_LR(image, 1.2); 

    % 从低分辨率图像中提取7x7的图像块
    lr_patches = Cutting(lr, 7); 

    % 获取提取的图像块的数量和大小
    [patch_num, size_pow] = size(lr_patches); 

    % 检查是否超出预定义的样本数量限制
    if patch_num + sign > 200000
        % 如果图像块数量超过200000，取前200000个
        sample(sign : 200000, :) = lr_patches(1 : 200000 - sign + 1, :); 
        break; % 退出循环
    else
        % 否则存储图像块
        sample(sign : sign + patch_num - 1, :) = lr_patches; 
        % 更新计数器
        sign = sign + patch_num; 
    end 
end

% 对样本进行特征提取
lr_features = Get_Feature(sample); 

% 对提取的特征进行聚类
Clusting(lr_features); 

toc; % 输出处理时间
%end
