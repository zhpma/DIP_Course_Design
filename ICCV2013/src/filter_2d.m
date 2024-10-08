function imgout = filter_2d(filter, img)
% FILTER_2D ��ͼ����ж�ά�˲�����
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
    
    nm = sm + 2 * bias;
    nn = sn + 2 * bias;
    
    % ��չ����ͼ��
    maps = padarray(f, [bias, bias], 'replicate', 'both');
    
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
    imgout = zeros(sm, sn);
    for i = 1 : sm
        for j = 1 : sn
            imgout(i, j) = result(i + bias, j + bias);
        end
    end

end
