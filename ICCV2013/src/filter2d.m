function [ imgout ] = filter2d(filter, img)
% FILTER2D ��ͼ����ж�ά�˲�����
%   ���������
%   - filter: �˲�������
%   - img: ����ͼ�����
%
%   ���������
%   - imgout: �˲�������ͼ��

    f = img;
    
    % ��ȡ����ͼ����˲����ĳߴ�
    [sm, sn] = size(f);
    [fm, fn] = size(filter);
    
    % ��ȡƫ����
    bias = fm - 1;
    
    % ��չ����ͼ��
    nm = sm + 2 * bias;
    nn = sn + 2 * bias;
    
    maps = zeros(nm, nn);
    result = zeros(nm, nn);
    
    % ������ͼ���Ƶ���չ���ͼ����
    for i = 1 : sm
        for j = 1 : sn
            maps(i + bias, j + bias) = f(i, j);
        end
    end
    
    % �˲�����
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
    
    % ��ȡ�˲����
    imgout = zeros(sm - fm + 1, sn - fn + 1);
    for i = 1 : sm - fm + 1
        for j = 1 : sn - fn + 1
            imgout(i, j) = result(i + bias + (fm - 1) / 2, j + bias + (fn - 1) / 2);
        end
    end

end
