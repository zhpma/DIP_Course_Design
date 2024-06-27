function SR_main( impath )
% SR_main 函数用于读取输入图像，进行超分辨率处理，并保存结果图像
% 输入:
%    impath - 输入图像的路径
% 输出:
%    无显式输出，但会保存处理后的高分辨率图像，并显示原始图像、低分辨率图像和高分辨率图像

    tic; % 开始计时

    % 读取输入图像
    f = imread(impath);
    
    % 将图像转换为双精度类型
    f = double(f);
    
    % 创建9x9的高斯滤波器
    win = zeros(9, 9);     
    center = (9 - 1) / 2 + 1;
    for i = 1 : 9
        for j = 1 : 9
            win(i,j) = exp(-((i - center)^2 + (j - center)^2) /(2 * 1.2^2)) / (2 * pi * 1.2^2);
        end
    end

    % 归一化滤波器
    win = win / sum(sum(win));
    
    % 获取图像尺寸
    [h, w, d] = size(f);
    
    if d == 1 % 单通道图像（灰度图像）
        % 应用高斯滤波器
        f = filter_2d(win, f);
        % 使用双三次插值生成低分辨率图像
        lr = bicubic(f, floor(h / 3), floor(w / 3));
        % 生成高分辨率图像
        hr = Generate_HR(lr);
    else % 多通道图像（彩色图像）
        % 分别对RGB通道应用高斯滤波器
        r = f(:, :, 1);
        r = filter_2d(win, r);
        g = f(:, :, 2);
        g = filter_2d(win, g);
        b = f(:, :, 3);
        b = filter_2d(win, b);
        
        % 分别对RGB通道应用双三次插值生成低分辨率图像
        r = bicubic(r, floor(h / 3), floor(w / 3));
        g = bicubic(g, floor(h / 3), floor(w / 3));
        b = bicubic(b, floor(h / 3), floor(w / 3));
        
        % 将低分辨率的RGB通道合并
        lr(:, :, 1) = r;
        lr(:, :, 2) = g;
        lr(:, :, 3) = b;
        
        % 将RGB图像转换为YCbCr
        temp = rgb2ycbcr(lr);
        % 取Y通道
        temp2 = temp(:, :, 1);
        % 生成高分辨率Y通道
        hr(:, :, 1) = Generate_HR(temp2);
        
        % 获取高分辨率Y通道尺寸
        [nh, nw] = size(hr(:, :, 1));
        % 分别对低分辨率的Cb和Cr通道应用双三次插值
        hr(:, :, 2) = bicubic(temp(:, :, 2), nh, nw);
        hr(:, :, 3) = bicubic(temp(:, :, 3), nh, nw);
        % 将YCbCr图像转换回RGB
        hr = ycbcr2rgb(hr);
    end

    % 显示原始图像
    figure, imshow(uint8(f));
    % 显示低分辨率图像
    figure, imshow(lr);
    % 显示高分辨率图像
    figure, imshow(hr);
    % 保存低分辨率图像
    imwrite(uint8(f), '1.png');

    % 保存中分辨率图像
    imwrite(lr, '2.png');

    % 保存高分辨率图像
    imwrite(hr, '3.png');



    imwrite(hr, '../Set14/testnew_sr.bmp');
    
    toc; % 结束计时
end
