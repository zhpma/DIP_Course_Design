import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

# 创建一个命令行参数解析器对象
parser = argparse.ArgumentParser(description='Train Super Resolution Models')
# 用于指定训练图像的裁剪尺寸，默认为88
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
# 用于指定超分辨率的放大因子，默认为4
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
# 用于指定训练的轮数，默认为100
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')

if __name__ == '__main__':
    # 解析命令行参数并将结果存储在变量opt中
    opt = parser.parse_args()

    # 从opt中获取crop_size、upscale_factor和num_epochs的值，并分别赋给对应的变量
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    # 创建训练数据集对象TrainDatasetFromFolder，指定数据集路径、裁剪尺寸和放大因子
    train_set = TrainDatasetFromFolder('data/VOC2012/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # 创建验证数据集对象ValDatasetFromFolder，指定数据集路径和放大因子
    val_set = ValDatasetFromFolder('data/VOC2012/val', upscale_factor=UPSCALE_FACTOR)
    # 创建训练数据加载器，指定数据集对象、工作线程数、批量大小和是否打乱数据顺序
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    # 创建验证数据加载器，指定数据集对象、工作线程数、批量大小和是否打乱数据顺序
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    # 创建生成器模型对象Generator，指定放大因子
    netG = Generator(UPSCALE_FACTOR)
    # 输出生成器模型参数的数量
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    # 创建生成器损失函数对象GeneratorLoss
    netD = Discriminator()
    # 输出判别器模型参数的数量
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    # 创建生成器损失函数对象GeneratorLoss
    generator_criterion = GeneratorLoss()

    # GPU如果可用的话，将生成器模型、判别器模型和生成器损失函数移动到GPU上进行计算
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    # 创建生成器和判别器的优化器对象，用于更新模型参数
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    # 创建一个字典用于存储训练过程中的判别器和生成器的损失、分数和评估指标结果(信噪比和相似性)
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        # 创建训练数据的进度条
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()  # 将生成器设置为训练模式
        netD.train()  # 将判别器设置为训练模式
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            # (1) Update D network: maximize D(x)-1-D(G(z))
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)  # 通过生成器生成伪图像

            # 清除判别器的梯度
            netD.zero_grad()
            # 通过判别器对真实图像进行前向传播，并计算其输出的平均值
            real_out = netD(real_img).mean()
            # 通过判别器对伪图像进行前向传播，并计算其输出的平均值
            fake_out = netD(fake_img).mean()
            # 计算判别器的损失
            d_loss = 1 - real_out + fake_out
            # 在判别器网络中进行反向传播，并保留计算图以进行后续优化步骤
            d_loss.backward(retain_graph=True)
            # 利用优化器对判别器网络的参数进行更新
            optimizerD.step()

            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            netG.zero_grad()
            # The two lines below are added to prevent runtime error in Google Colab
            # 通过生成器对输入图像（z）进行生成，生成伪图像（fake_img）
            fake_img = netG(z)
            # 通过判别器对伪图像进行前向传播，并计算其输出的平均值
            fake_out = netD(fake_img).mean()
            ##
            # 计算生成器的损失，包括对抗损失、感知损失、图像损失和TV损失
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            # 在生成器网络中进行反向传播，计算生成器的梯度
            g_loss.backward()

            # 再次通过生成器对输入图像（z）进行生成，得到新的伪图像（fake_img）
            fake_img = netG(z)
            # 通过判别器对新的伪图像进行前向传播，并计算其输出的平均值
            fake_out = netD(fake_img).mean()
            # 利用优化器对生成器网络的参数进行更新
            optimizerG.step()

            # loss for current batch before optimization
            # 累加当前批次生成器的损失值乘以批次大小，用于计算平均损失
            running_results['g_loss'] += g_loss.item() * batch_size
            # 累加当前批次判别器的损失值乘以批次大小，用于计算平均损失
            running_results['d_loss'] += d_loss.item() * batch_size
            # 累加当前批次真实图像在判别器的输出得分乘以批次大小，用于计算平均得分
            running_results['d_score'] += real_out.item() * batch_size
            # 累加当前批次伪图像在判别器的输出得分乘以批次大小，用于计算平均得分
            running_results['g_score'] += fake_out.item() * batch_size

            # 更新训练进度条的描述信息
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        # 创建用于保存训练结果的目录
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            # 遍历验证数据集(低分辨率图 恢复的高分辨率图 高分辨率图)
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()

                # 生成超分辨率图像
                sr = netG(lr)

                # 计算批量图像的均方误差
                batch_mse = ((sr - hr) ** 2).data.mean()
                # 累加均方误差
                valing_results['mse'] += batch_mse * batch_size
                # 计算批量图像的结构相似度指数
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                # 累加结构相似度指数
                valing_results['ssims'] += batch_ssim * batch_size
                # 计算平均峰值信噪比
                valing_results['psnr'] = 10 * log10(
                    (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                # 计算平均结构相似度指数
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                # 更新训练进度条的描述信息
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

                val_images.extend(
                    # 将图像应用转换函数，并添加到验证图像列表
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])

            # 将验证图像列表堆叠为张量
            val_images = torch.stack(val_images)
            # 将堆叠后的张量分割为多个小块，每个小块包含15张图像
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            # 创建进度条，并设置描述为“[saving training results]”
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                # 将小块中的图像创建为一个网格，每行显示3张图像，图像之间有5个像素的间隔
                image = utils.make_grid(image, nrow=3, padding=5)
                # 将网格图像保存为文件，文件名包含epoch和index信息
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        # save model parameters
        # 将判别器和生成器的参数保存到指定文件
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))

        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            # 创建一个DataFrame对象，用于存储训练结果数据
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            # 将DataFrame对象保存为CSV文件
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')

