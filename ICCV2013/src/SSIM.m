function [ output ] = SSIM( input_img1, input_img2 )
% SSIM 函数用于计算两个图像之间的结构相似性指数 (SSIM)
% 输入:
%    input_img1 - 输入图像1
%    input_img2 - 输入图像2
% 输出:
%    output - 两个图像之间的SSIM值

    % 获取输入图像的尺寸
    flag = size(input_img1);
    
    % 如果图像是彩色图像，转换为灰度图像（取Y通道）
    if numel(flag) > 2
        input_img1 = rgb2ycbcr(input_img1);
        input_img1 = input_img1(:,:,1);
        input_img2 = rgb2ycbcr(input_img2);
        input_img2 = input_img2(:,:,1);
    end
    
    % 将图像转换为双精度类型
    X = double(input_img1);
    Y = double(input_img2);
    
    % 初始化11x11的高斯滤波器
    w = zeros(11, 11);
    center = (11 - 1) / 2 + 1;
    for i = 1 : 11
        for j = 1 : 11
            w(i,j) = exp(-((i - center)^2 + (j - center)^2) /(2 * 1.5^2)) / (2 * pi * 1.5^2);
        end
    end
    w = w / sum(sum(w)); % 归一化滤波器
    
    % 定义常量
    k1 = 0.01;
    k2 = 0.03;
    L = 255;
    c1 = (k1 * L)^2;
    c2 = (k2 * L)^2;
    
    % 计算加权均值和方差
    ua = filter2d(w, X);
    ub = filter2d(w, Y);
    ua_sq = ua .* ua;
    ub_sq = ub .* ub;
    ua_ub = ua .* ub;
    siga_sq = filter2d(w, X .* X) - ua_sq;
    sigb_sq = filter2d(w, Y .* Y) - ub_sq;
    sigab = filter2d(w, X .* Y) - ua_ub;
    
    % 计算并返回SSIM值
    output = mean2(((2 * ua_ub + c1) .* (2 * sigab + c2)) ./ ((ua_sq + ub_sq + c1) .* (siga_sq + sigb_sq + c2)));

end
