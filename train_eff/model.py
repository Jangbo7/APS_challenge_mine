import torch
import torch.nn as nn
from torchvision import models


def get_efficientnetv2(num_classes=17, pretrained=True):
    """
    获取 EfficientNetV2 模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: EfficientNetV2 模型
    """
    # 加载 EfficientNetV2-S 模型
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    
    # 修改分类器
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model


def get_efficientnetv2_m(num_classes=17, pretrained=True):
    """
    获取 EfficientNetV2-M 模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: EfficientNetV2-M 模型
    """
    weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_m(weights=weights)
    
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model


def get_efficientnetv2_l(num_classes=17, pretrained=True):
    """
    获取 EfficientNetV2-L 模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: EfficientNetV2-L 模型
    """
    weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_l(weights=weights)
    
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model


class EfficientNetV2Classifier(nn.Module):
    """
    EfficientNetV2 分类器包装类
    """
    def __init__(self, num_classes=17, model_type='s', pretrained=True, dropout=0.2):
        super(EfficientNetV2Classifier, self).__init__()
        
        if model_type == 's':
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_v2_s(weights=weights)
        elif model_type == 'm':
            weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_v2_m(weights=weights)
        elif model_type == 'l':
            weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_v2_l(weights=weights)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 获取特征维度
        num_features = self.backbone.classifier[1].in_features
        
        # 替换分类器
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """获取特征向量（用于特征提取或可视化）"""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def build_model(config):
    """
    根据配置构建模型
    
    Args:
        config: 配置对象
    
    Returns:
        model: 构建好的模型
    """
    model = EfficientNetV2Classifier(
        num_classes=config.NUM_CLASSES,
        model_type='s',  # 默认使用 EfficientNetV2-S
        pretrained=True,
        dropout=0.2
    )
    return model
