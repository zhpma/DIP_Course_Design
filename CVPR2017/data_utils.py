from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    # 判断文件名是否是图像文件
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    # 计算可用的裁剪尺寸
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    # 训练集的高分辨率图像转换
    return Compose([
        RandomCrop(crop_size),  # 随机裁剪图像到指定尺寸
        ToTensor(),  # 将图像转换为张量
    ])


def train_lr_transform(crop_size, upscale_factor):
    # 训练集的低分辨率图像转换
    return Compose([
        ToPILImage(),  # 将张量转换为PIL图像
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC), # 将图像缩放到指定尺寸，使用双三次插值方法
        ToTensor()   # 将图像转换为张量
    ])


def display_transform():
    # 显示图像的转换
    return Compose([
        ToPILImage(),   # 将张量转换为PIL图像
        Resize(400),  # 将图像调整大小为400x400
        CenterCrop(400), # 对图像进行中心裁剪为400x400
        ToTensor()   # 将图像转换为张量
    ])

#  加载训练集中的图像数据
class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        # 获取目录中的所有图像文件名，并使用is_image_file函数来筛选出图像文件
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        # 计算可用的裁剪尺寸
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        # 分别创建高分辨率和低分辨率图像的转换操作
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        # 获取给定索引的图像数据
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        # 返回数据集的大小（图像数量）
        return len(self.image_filenames)

#  加载验证集中的图像数据 同训练集
class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        # 打开高分辨率图像文件
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size  # 获取图像的宽度和高度
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)   # 计算可用的裁剪尺寸
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)  # 缩放图像为低分辨率图像
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)  # 缩放图像为高分辨率图像
        hr_image = CenterCrop(crop_size)(hr_image)    # 对高分辨率图像进行中心裁剪
        lr_image = lr_scale(hr_image)   # 缩放得到低分辨率图像
        hr_restore_img = hr_scale(lr_image) # 缩放得到还原后的高分辨率图像
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

#  加载测试集中的图像数据 同训练集
class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'  # 构建低分辨率图像文件路径
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/' # 构建高分辨率图像文件路径
        self.upscale_factor = upscale_factor
        # 获取两个路径下的图像文件名，并保存在lr_filenames和hr_filenames列表中。
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        # 获取给定索引的图像数据
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])  # 打开低分辨率图像文件
        w, h = lr_image.size  # 获取低分辨率图像的宽度和高度
        hr_image = Image.open(self.hr_filenames[index])  # 打开高分辨率图像文件
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC) # 缩放高分辨率图像
        hr_restore_img = hr_scale(lr_image)  # 缩放得到还原后的高分辨率图像
        # 将图像文件名、低分辨率图像、还原后的高分辨率图像和原始高分辨率图像转换为张量并返回
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)

