function [ output_img ] = bicubic( input_img, height, width )
%BICUBIC 使用双三次插值法对图像进行缩放
%   输入参数：
%   - input_img: 输入图像
%   - height: 输出图像的高度
%   - width: 输出图像的宽度
%
%   输出参数：
%   - output_img: 缩放后的图像

    [m, n] = size(input_img);  % 获取输入图像的尺寸
    
    f = double(input_img);  % 将输入图像转换为双精度类型
    
    x_rate = height/m;  % 计算高度缩放比例
    y_rate = width/n;   % 计算宽度缩放比例
    
    output_img = zeros(height, width);  % 初始化输出图像
    
    for i = 1 : height
        for j = 1 : width
            sx = i/x_rate;  % 计算目标像素对应的源图像位置x坐标
            sy = j/y_rate;  % 计算目标像素对应的源图像位置y坐标
            
            si = floor(sx);  % 获取源图像x坐标的整数部分
            sj = floor(sy);  % 获取源图像y坐标的整数部分
            
            u = sx - si;  % 获取源图像x坐标的小数部分
            v = sy - sj;  % 获取源图像y坐标的小数部分
            
            % 确保索引在合理范围内
            if (si + 1) > m
                si = m - 2;
            end
            
            if (si + 2) > m
                si = m - 2;
            end
            
            if (si - 1) < 1
                si = 2;
            end
            
            if (sj + 1) > n
                sj = n - 2;
            end
            
            if (sj + 2) > n
                sj = n - 2;
            end
            
            if (sj - 1) < 1
                sj = 2;
            end
            
            % 双三次插值的权重矩阵
            A = [W(1 + u) W(u) W(1 - u) W(2 - u)];
            C = [W(1 + v); W(v); W(1 - v); W(2 - v)];
            B = [f(si - 1, sj - 1) f(si - 1, sj) f(si - 1, sj + 1) f(si - 1, sj + 2);
                 f(si, sj - 1) f(si, sj) f(si, sj + 1) f(si, sj + 2);
                 f(si + 1, sj - 1) f(si + 1, sj) f(si + 1, sj + 1) f(si + 1, sj + 2);
                 f(si + 2, sj - 1) f(si + 2, sj) f(si + 2, sj + 1) f(si + 2, sj + 2);];

            % 计算插值后的像素值
            output_img(i,j) = (A * B * C);
           
        end
    end
    
    output_img = uint8(output_img);  % 将输出图像转换为8位无符号整数类型
            
end
