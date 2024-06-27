function  lr = HR_To_LR( hr, sigma)
% HR_To_LR 将高分辨率图像转换为低分辨率图像
% 输入:
%    hr - 高分辨率图像
%    sigma - 高斯滤波器的标准差
% 输出:
%    lr - 低分辨率图像

    % 将图像转换为双精度类型
    hr = double(hr);
    
    % 获取图像尺寸
    [h, w, d] = size(hr);
    
    % 确保高分辨率图像的高度和宽度可以被3整除
    hr_height = h - mod(h, 3);
    hr_width = w - mod(w, 3);
    
    % 截取调整后的高分辨率图像
    new_hr = hr(1:hr_height, 1:hr_width, 1:d);
    
    % 计算低分辨率图像的高度和宽度
    lr_height = hr_height / 3;
    lr_width = hr_width / 3;
    
    % 计算高斯核的大小
    kernel_size = ceil(sigma * 3) * 2 + 1;
    
    % 创建高斯滤波器
    win = zeros(kernel_size, kernel_size);    
    center = (kernel_size - 1) / 2 + 1;
    for i = 1 : kernel_size
        for j = 1 : kernel_size
            win(i,j) = exp(-((i - center)^2 + (j - center)^2) / (2 * sigma^2)) / (2 * pi * sigma^2);
        end
    end
    
    % 归一化滤波器
    win = win / sum(sum(win));
    
    % 如果图像是灰度图像
    if d == 1
        temp_img = filter_2d(win, new_hr);
    % 如果图像是彩色图像
    elseif d == 3
        new_hr = rgb2ycbcr(new_hr); % 将图像转换为YCbCr颜色空间
        new_hr = new_hr(:, :, 1); % 只取Y通道
        temp_img = filter_2d(win, new_hr);
    end
    
    % 使用双三次插值将图像降采样为低分辨率图像
    lr = bicubic(temp_img, lr_height, lr_width);
end
