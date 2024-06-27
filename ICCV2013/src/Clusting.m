function Clusting( feature )
% Clusting 函数
%   该函数用于对输入的特征数据进行聚类，并将聚类中心保存到文件中。
%   输入参数:
%       feature - 输入的特征数据矩阵，每一行代表一个数据点。

    % 设置k-means算法的选项
    opts = statset('Display', 'iter', ... % 显示迭代信息
                   'MaxIter', 1000, ...   % 最大迭代次数
                   'UseParallel', 1);     % 使用并行计算

    % 进行k-means聚类，分成512个簇
    [IDX, C] = kmeans(feature, 512, 'options', opts);

    % 将聚类中心保存到文件 center.mat 中
    save('../lib/center.mat', 'C');
end
