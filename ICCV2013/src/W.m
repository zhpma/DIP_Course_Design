function [ output ] = W( input )
% W 函数用于根据输入值计算输出值
% 输入:
%    input - 输入值
% 输出:
%    output - 输出值

    % 计算输入值的绝对值
    x = abs(input);
    
    % 根据不同的区间计算输出值
    if (0 <= x && x <= 1)
        % 如果x在0到1之间，使用第一个公式计算输出值
        output = double(1.5 * (x^3) - 2.5 * (x^2) + 1);
    elseif (1 < x && x <= 2)
        % 如果x在1到2之间，使用第二个公式计算输出值
        output = double(-0.5 * (x^3) + 2.5 * (x^2) - 4 * x + 2);
    else
        % 如果x不在0到2之间，输出值为0
        output = 0;
    end

end
