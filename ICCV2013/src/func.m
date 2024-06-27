function [output] = func(impath)
% func 从输入路径读取图像并计算其PSNR和SSIM
%   此函数读取图像，进行下采样和上采样，并计算PSNR和SSIM值。

    % 读取输入图像
    f = imread(impath);
    
    % 获取图像的尺寸
    flag = size(f);
    
    % 初始化输出数组
    output = zeros(1, 2);
    
    % 显示原始图像
    figure, imshow(f);
    
    % 如果图像是彩色图像
    if numel(flag) > 2
        r = f(:, :, 1); % 获取红色通道
        g = f(:, :, 2); % 获取绿色通道
        b = f(:, :, 3); % 获取蓝色通道
    
        [m, n] = size(r); % 获取图像的尺寸
    
        % 将每个通道下采样到原始尺寸的三分之一
        r = bicubic(r, round(m / 3), round(n / 3));
        g = bicubic(g, round(m / 3), round(n / 3));
        b = bicubic(b, round(m / 3), round(n / 3));

        % 将下采样后的通道重新组合成一个图像
        temp(:, :, 1) = r;
        temp(:, :, 2) = g;
        temp(:, :, 3) = b;
        
        % 显示下采样后的图像
        figure, imshow(temp);
        
        % 将每个通道上采样回原始尺寸
        input2(:, :, 1) = bicubic(r, m, n);
        input2(:, :, 2) = bicubic(g, m, n);
        input2(:, :, 3) = bicubic(b, m, n);

        % 显示上采样后的图像
        figure, imshow(input2);
        
        % 计算原始图像和上采样图像的PSNR和SSIM
        output(1) = PSNR(f, input2);
        output(2) = SSIM(f, input2);
        
    % 如果图像是灰度图像
    else
        [m, n] = size(f); % 获取图像的尺寸
        
        % 将图像下采样到原始尺寸的三分之一
        input1 = bicubic(f, round(m / 3), round(n / 3));
        figure, imshow(input1); % 显示下采样后的图像
        
        % 将图像上采样回原始尺寸
        input2 = bicubic(input1, m, n);
        figure, imshow(input2); % 显示上采样后的图像
        
        % 计算原始图像和上采样图像的PSNR和SSIM
        output(1, 1) = PSNR(f, input2);
        output(1, 2) = SSIM(f, input2);
    end
end
