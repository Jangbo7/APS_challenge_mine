import torch
import torch.nn as nn
import torch.hub
import os
from torchvision import models


def _adapt_stem_conv_in_channels(backbone: nn.Module, in_channels: int = 3, model_type='efficientnetv2'):
    """
    将模型首层卷积从3通道适配到指定的通道数。
    预训练权重迁移策略：
    - 前3通道保留原权重
    - 新增通道使用原3通道均值初始化
    
    Args:
        backbone: 模型backbone
        in_channels: 目标输入通道数
        model_type: 模型类型('efficientnetv2' 或 'convnext')
    
    Returns:
        修改后的backbone
    """
    if in_channels == 3:
        return backbone
    
    # 找到第一层卷积
    old_conv = None
    if model_type.startswith('efficientnetv2'):
        old_conv = backbone.features[0][0]  # EfficientNetV2 结构
    elif model_type.startswith('convnext'):
        old_conv = backbone.features[0]  # ConvNeXt 结构
    
    if old_conv is None or not isinstance(old_conv, nn.Conv2d):
        return backbone
    
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None),
        padding_mode=old_conv.padding_mode,
    )
    
    with torch.no_grad():
        # 原始权重: [out_c, 3, k, k]
        new_conv.weight[:, :3, :, :] = old_conv.weight
        if in_channels > 3:
            mean_w = old_conv.weight.mean(dim=1, keepdim=True)  # [out_c, 1, k, k]
            repeat_n = in_channels - 3
            new_conv.weight[:, 3:, :, :] = mean_w.repeat(1, repeat_n, 1, 1)
        else:
            # 极端情况: in_channels < 3
            new_conv.weight.copy_(old_conv.weight[:, :in_channels, :, :])
        
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    
    # 替换第一层卷积
    if model_type.startswith('efficientnetv2'):
        backbone.features[0][0] = new_conv
    elif model_type.startswith('convnext'):
        backbone.features[0] = new_conv
    
    return backbone


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


def get_convnext_tiny(num_classes=17, pretrained=True):
    """
    获取 ConvNeXt Tiny 模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: ConvNeXt Tiny 模型
    """
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)
    
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, num_classes)
    
    return model


def get_convnext_small(num_classes=17, pretrained=True):
    """
    获取 ConvNeXt Small 模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: ConvNeXt Small 模型
    """
    weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_small(weights=weights)
    
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, num_classes)
    
    return model


def get_convnext_base(num_classes=17, pretrained=True):
    """
    获取 ConvNeXt Base 模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: ConvNeXt Base 模型
    """
    weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_base(weights=weights)
    
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, num_classes)
    
    return model


def get_convnext_large(num_classes=17, pretrained=True):
    """
    获取 ConvNeXt Large 模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: ConvNeXt Large 模型
    """
    weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_large(weights=weights)
    
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, num_classes)
    
    return model


class ModelClassifier(nn.Module):
    """
    通用模型分类器包装类，支持 EfficientNetV2 和 ConvNeXt
    支持可变的输入通道数（3通道或5通道）
    """
    def __init__(self, num_classes=17, model_type='efficientnetv2_s', pretrained=True, dropout=0.2, in_channels=3):
        super(ModelClassifier, self).__init__()
        self.in_channels = in_channels
        
        if model_type.startswith('efficientnetv2'):
            sub_type = model_type.split('_')[-1]  # 's', 'm', 'l'
            if sub_type == 's':
                weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = models.efficientnet_v2_s(weights=weights)
                classifier_in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(p=dropout, inplace=True),
                    nn.Linear(classifier_in_features, num_classes)
                )
            elif sub_type == 'm':
                weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = models.efficientnet_v2_m(weights=weights)
                classifier_in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(p=dropout, inplace=True),
                    nn.Linear(classifier_in_features, num_classes)
                )
            elif sub_type == 'l':
                weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = models.efficientnet_v2_l(weights=weights)
                classifier_in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(p=dropout, inplace=True),
                    nn.Linear(classifier_in_features, num_classes)
                )
            else:
                raise ValueError(f"Unknown EfficientNetV2 type: {sub_type}")
            
            # 适配输入通道数
            if in_channels != 3:
                self.backbone = _adapt_stem_conv_in_channels(self.backbone, in_channels=in_channels, model_type='efficientnetv2')
                
        elif model_type.startswith('convnext'):
            sub_type = model_type.split('_')[-1]  # 'tiny', 'small', 'base', 'large'
            if sub_type == 'tiny':
                weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = models.convnext_tiny(weights=weights)
            elif sub_type == 'small':
                weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = models.convnext_small(weights=weights)
            elif sub_type == 'base':
                weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = models.convnext_base(weights=weights)
            elif sub_type == 'large':
                weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = models.convnext_large(weights=weights)
            else:
                raise ValueError(f"Unknown ConvNeXt type: {sub_type}")
            
            # 适配输入通道数
            if in_channels != 3:
                self.backbone = _adapt_stem_conv_in_channels(self.backbone, in_channels=in_channels, model_type='convnext')
            
            classifier_in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(classifier_in_features, num_classes)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """获取特征向量（用于特征提取或可视化）"""
        if hasattr(self.backbone, 'features'):
            # EfficientNetV2
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        elif hasattr(self.backbone, 'avgpool'):
            # ConvNeXt
            x = self.backbone.avgpool(self.backbone.features(x))
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
    # 设置预训练权重的下载缓存目录
    if config.PRETRAINED_WEIGHTS_DIR is not None:
        os.makedirs(config.PRETRAINED_WEIGHTS_DIR, exist_ok=True)
        torch.hub.set_dir(config.PRETRAINED_WEIGHTS_DIR)
    
    # 根据是否使用频率通道确定输入通道数
    in_channels = 5 if getattr(config, 'USE_FREQ_CHANNELS', False) else 3
    
    model = ModelClassifier(
        num_classes=config.NUM_CLASSES,
        model_type=config.MODEL_TYPE,
        pretrained=True,
        dropout=0.2,
        in_channels=in_channels
    )
    return model
