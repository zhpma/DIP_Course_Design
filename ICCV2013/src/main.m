function main( )
% main 函数用于计算一组图像的PSNR和SSIM值，并输出这些值及其平均值
% 实际使用bicubic进行上采样和下采样
% 输入:
%    无
% 输出:
%    显示每个图像的PSNR和SSIM值，以及它们的平均值

    % 初始化PSNR和SSIM的存储数组
    psnr = zeros(1, 14);
    ssim = zeros(1, 14);
    
    % 对每张图像进行处理，并获取其PSNR和SSIM值
    temp = func('../Set14/baboon.bmp');
    psnr(1) = temp(1);
    ssim(1) = temp(2);
    
    temp = func('../Set14/barbara.bmp');
    psnr(2) = temp(1);
    ssim(2) = temp(2);
    
    temp = func('../Set14/bridge.bmp');
    psnr(3) = temp(1);
    ssim(3) = temp(2);
    
    temp = func('../Set14/coastguard.bmp');
    psnr(4) = temp(1);
    ssim(4) = temp(2);
    
    temp = func('../Set14/comic.bmp');
    psnr(5) = temp(1);
    ssim(5) = temp(2);
    
    temp = func('../Set14/face.bmp');
    psnr(6) = temp(1);
    ssim(6) = temp(2);
    
    temp = func('../Set14/flowers.bmp');
    psnr(7) = temp(1);
    ssim(7) = temp(2);
    
    temp = func('../Set14/foreman.bmp');
    psnr(8) = temp(1);
    ssim(8) = temp(2);
    
    temp = func('../Set14/lenna.bmp');
    psnr(9) = temp(1);
    ssim(9) = temp(2);
    
    temp = func('../Set14/man.bmp');
    psnr(10) = temp(1);
    ssim(10) = temp(2);
    
    temp = func('../Set14/monarch.bmp');
    psnr(11) = temp(1);
    ssim(11) = temp(2);
    
    temp = func('../Set14/pepper.bmp');
    psnr(12) = temp(1);
    ssim(12) = temp(2);
    
    temp = func('../Set14/ppt3.bmp');
    psnr(13) = temp(1);
    ssim(13) = temp(2);
    
    temp = func('../Set14/zebra.bmp');
    psnr(14) = temp(1);
    ssim(14) = temp(2);
    
    % 显示每个图像的PSNR值
    disp(psnr);
    % 显示每个图像的SSIM值
    disp(ssim);
    % 计算并显示PSNR的平均值
    disp(mean(psnr));
    % 计算并显示SSIM的平均值
    disp(mean(ssim));
    
end
