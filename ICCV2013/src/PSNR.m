function [ output ] = PSNR( input_img1, input_img2 )
% PSNR 函数用于计算两个图像之间的峰值信噪比 (PSNR)
% 输入:
%    input_img1 - 输入图像1
%    input_img2 - 输入图像2
% 输出:
%    output - 两个图像之间的PSNR值

    % 获取输入图像的尺寸
    flag = size(input_img1);
    
    % 如果图像是彩色图像，将其转换为灰度图像（取Y通道）
    if numel(flag) > 2
        input_img1 = rgb2ycbcr(input_img1);
        input_img1 = input_img1(:,:,1);
        input_img2 = rgb2ycbcr(input_img2);
        input_img2 = input_img2(:,:,1);
    end

    % 将图像转换为双精度类型
    X = double(input_img1);
    Y = double(input_img2);
    
    % 获取图像尺寸
    [M, N] = size(X);
    
    % 初始化累加和
    sum = 0;
    
    % 计算两个图像之间的均方误差 (MSE)
    for i = 1 : M
        for j = 1 : N
            sum = sum + (X(i,j) - Y(i,j))^2;
        end
    end
    
    % 计算MSE
    MSE = double(sum / (M * N));
    
    % 计算并返回PSNR值
    output = 20 * log10(double(255 / (MSE^0.5)));

end
