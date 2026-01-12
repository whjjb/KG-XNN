# kgxnn/data_loaders/cifar.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义ImageNet数据集的均值和标准差，用于图像标准化
# 这些值是针对RGB三个通道计算得出的，用于与ImageNet预训练模型兼容
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB三个通道的均值
IMAGENET_STD  = [0.229, 0.224, 0.225]  # RGB三个通道的标准差

def get_loaders(
    data_root: str = "./data",      # 数据存储根目录路径，默认为当前目录下的data文件夹
    batch_size: int = 128,          # 每个批次的样本数量，默认为128
    num_workers: int = 4,           # 数据加载使用的进程数，默认为4
    img_size: int = 224,            # 输入图像的目标尺寸，默认为224x224
):
    """
    返回 CIFAR-100 的 train/val DataLoader。
    - 我们把 32x32 的 CIFAR 图缩放到 224x224，以适配 ImageNet 预训练的 ResNet。

    参数:
        data_root (str): 数据集存储路径
        batch_size (int): 批次大小
        num_workers (int): 多进程数据加载的工作进程数
        img_size (int): 图像调整后的目标尺寸

    返回:
        tuple: (train_loader, val_loader, num_classes)
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_classes: 类别数量(对于CIFAR-100为100)
    """

    # 定义训练集的数据增强和预处理流程
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),        # 将图像调整为目标尺寸
        transforms.RandomHorizontalFlip(),              # 随机水平翻转，增加数据多样性
        transforms.ToTensor(),                          # 转换PIL图像为tensor格式
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # 使用ImageNet的均值和标准差进行标准化
    ])

    # 定义验证集的预处理流程（不包含数据增强）
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),        # 将图像调整为目标尺寸
        transforms.ToTensor(),                          # 转换PIL图像为tensor格式
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # 使用ImageNet的均值和标准差进行标准化
    ])

    # 创建训练数据集对象，使用CIFAR-100训练集
    train_set = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_tf)

    # 创建验证数据集对象，使用CIFAR-100测试集
    val_set   = datasets.CIFAR100(root=data_root, train=False, download=True, transform=val_tf)

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,               # 打乱训练数据顺序
        num_workers=num_workers,
        pin_memory=True             # 锁页内存加速GPU传输
    )

    # 创建验证数据加载器
    val_loader   = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,              # 验证数据不打乱顺序
        num_workers=num_workers,
        pin_memory=True             # 锁页内存加速GPU传输
    )

    # 返回训练和验证数据加载器以及类别数量(100)
    return train_loader, val_loader, 100
