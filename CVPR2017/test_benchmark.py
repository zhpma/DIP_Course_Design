import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_150.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

# 保存每个测试数据集的结果
results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

# 创建一个 Generator 对象
model = Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()
# 加载训练好的模型参数
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

# 加载测试数据集
test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
# 创建一个用于 test_loader 的 tqdm 进度条
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

# 测试结果输出路径
out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

for image_name, lr_image, hr_restore_img, hr_image in test_bar:
    # 由于 image_name 是一个包含单个元素的列表，所以将其取出
    image_name = image_name[0]
    # 将 lr_image 转换为 Variable 对象，并设置 volatile=True
    # volatile=True 表示不会计算梯度，这在推理阶段通常是需要的
    lr_image = Variable(lr_image, volatile=True)
    hr_image = Variable(hr_image, volatile=True)
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()

    # 生成超分变率图像
    sr_image = model(lr_image)

    mse = ((hr_image - sr_image) ** 2).data.mean()
    # 计算峰值信噪比（Peak Signal-to-Noise Ratio）
    psnr = 10 * log10(255 ** 2 / mse)
    # 计算结构相似性指数（Structural Similarity Index）
    # 使用 pytorch_ssim 库中的 ssim 函数计算 SSIM
    ssim = pytorch_ssim.ssim(sr_image, hr_image).data[0]

    # 创建一个包含三张图像的张量，分别是原始恢复的高分辨率图像、原始高分辨率图像和生成的超分辨率图像
    # 将每张图像应用 display_transform() 转换，并通过 squeeze(0) 去除批次维度
    test_images = torch.stack(
        [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
         display_transform()(sr_image.data.cpu().squeeze(0))])

    # 使用 make_grid 函数将三张图像拼接成一张大图像
    # nrow=3 表示每行显示 3 张图像，padding=5 表示图像之间的间距为 5
    image = utils.make_grid(test_images, nrow=3, padding=5)

    # 使用 save_image 函数将合成的图像保存到指定路径
    utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                     image_name.split('.')[-1], padding=5)

    # 将对应数据集的PSNR和SSIM保存到对应的字典当中
    results[image_name.split('_')[0]]['psnr'].append(psnr)
    results[image_name.split('_')[0]]['ssim'].append(ssim)

# 最终结果保存路径
out_path = 'statistics/'
saved_results = {'psnr': [], 'ssim': []}

# 遍历 results 字典中的每个值
for item in results.values():
    # 获取 PSNR 和 SSIM 的列表
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])

    # 如果列表为空，将 PSNR 和 SSIM 设置为 'No data'
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        # 如果列表不为空，计算 PSNR 和 SSIM 的均值
        psnr = psnr.mean()
        ssim = ssim.mean()

    # 将计算得到的 PSNR 和 SSIM 添加到 saved_results 字典的相应列表中
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

# 创建一个 DataFrame 对象，使用 saved_results 字典作为数据，以 results.keys() 作为列标签
data_frame = pd.DataFrame(saved_results, results.keys())
# 将 DataFrame 对象保存为 CSV 文件
# 文件路径由 out_path、'srf_'、UPSCALE_FACTOR 值和 '_test_results.csv' 组成
# index_label='DataSet' 表示使用 'DataSet' 作为索引标签
data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')

