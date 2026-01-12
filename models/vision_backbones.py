# kgxnn/models/vision_backbones.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Embed(nn.Module):
    """
    ResNet-50 作为特征提取器：
    - 输出向量维度 2048
    - 去掉最后的全连接层（fc）
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()  # 调用父类 nn.Module 的初始化方法

        # 根据 pretrained 参数决定是否使用预训练权重
        if pretrained:
            # 使用 ImageNet 上预训练的 ResNet50 权重
            # ResNet50_Weights.IMAGENET1K_V2 是官方推荐的预训练权重版本
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = resnet50(weights=weights)
        else:
            # 不使用预训练权重，从随机初始化开始训练
            self.backbone = resnet50(weights=None)

        # 移除 ResNet50 最后的全连接分类层
        # nn.Identity() 是一个恒等映射，输入即输出，相当于移除了原 fc 层
        # 这样网络就只保留特征提取功能，不进行分类
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        # 前向传播函数
        # 输入 x 经过修改后的 ResNet50 网络
        # 输出为 [B, 2048] 的特征向量，其中 B 是批次大小
        # 2048 是 ResNet50 最后一层特征图展平后的维度
        return self.backbone(x)
