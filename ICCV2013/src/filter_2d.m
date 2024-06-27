function imgout = filter_2d(filter, img)
% FILTER_2D 对图像进行二维滤波操作
%   输入参数：
%   - filter: 滤波器矩阵
%   - img: 输入图像矩阵
%
%   输出参数：
%   - imgout: 滤波后的输出图像

    f = img;
    
    % 获取输入图像和滤波器的尺寸
    [sm, sn] = size(f);
    [fm, fn] = size(filter);
    
    % 获取偏移量
    bias = fm - 1;
    
    nm = sm + 2 * bias;
    nn = sn + 2 * bias;
    
    % 扩展输入图像
    maps = padarray(f, [bias, bias], 'replicate', 'both');
    
    % 滤波操作
    for i = (1 + bias / 2) : (nm - bias / 2)
        for j = (1 + bias / 2) : (nn - bias / 2)
            sum = 0;
            for x = 1 : fm
                for y = 1 : fn
                    sum = sum + filter(x, y) * maps(i - bias / 2 + x - 1, j - bias / 2 + y - 1);
                end
            end
            result(i, j) = sum;
        end
    end
    
    % 获取滤波结果
    imgout = zeros(sm, sn);
    for i = 1 : sm
        for j = 1 : sn
            imgout(i, j) = result(i + bias, j + bias);
        end
    end

end
