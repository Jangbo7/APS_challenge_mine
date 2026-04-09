import torch
import torch.nn as nn
import torch.hub
import os
import sys
import builtins
from pathlib import Path
from torchvision import models
from config import Config
import glob

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm library not found. ConvNeXtV2 models will not be available.")

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def print(*args, **kwargs):
    file = kwargs.get('file', sys.stdout)
    encoding = getattr(file, 'encoding', None)
    if encoding:
        safe_args = []
        for arg in args:
            text = str(arg)
            safe_args.append(text.encode(encoding, errors='replace').decode(encoding, errors='replace'))
        return builtins.print(*safe_args, **kwargs)
    return builtins.print(*args, **kwargs)


def _find_last_linear(module: nn.Module):
    """在模块中查找最后一个 Linear 层。"""
    last_linear = None
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            last_linear = layer
    return last_linear


def _adapt_stem_conv_in_channels(backbone: nn.Module, in_channels: int = 3, model_type='efficientnetv2'):
    """
    将模型首层卷积从3通道适配到指定的通道数。
    预训练权重迁移策略：
    - 前3通道保留原权重
    - 新增通道使用原3通道均值初始化
    
    Args:
        backbone: 模型backbone
        in_channels: 目标输入通道数
        model_type: 模型类型('efficientnetv2', 'convnext', 或 'convnextv2')
    
    Returns:
        修改后的backbone
    """
    if in_channels == 3:
        return backbone
    
    # 找到第一层卷积
    old_conv = None
    stem_parent = None
    stem_key = None
    if model_type.startswith('efficientnetv2'):
        old_conv = backbone.features[0][0]  # EfficientNetV2 结构
        stem_parent = backbone.features[0]
        stem_key = 0
    elif model_type.startswith('convnext'):
        if hasattr(backbone, 'stem'):
            if isinstance(backbone.stem, nn.Sequential) and len(backbone.stem) > 0 and isinstance(backbone.stem[0], nn.Conv2d):
                old_conv = backbone.stem[0]
                stem_parent = backbone.stem
                stem_key = 0
            elif isinstance(backbone.stem, nn.Conv2d):
                old_conv = backbone.stem
                stem_parent = backbone
                stem_key = 'stem'
        elif hasattr(backbone, 'features'):
            first = backbone.features[0]
            if isinstance(first, nn.Conv2d):
                old_conv = first
                stem_parent = backbone.features
                stem_key = 0
            elif isinstance(first, nn.Sequential) and len(first) > 0 and isinstance(first[0], nn.Conv2d):
                old_conv = first[0]
                stem_parent = first
                stem_key = 0
    
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
    if stem_parent is not None:
        if isinstance(stem_key, int):
            stem_parent[stem_key] = new_conv
        else:
            setattr(stem_parent, stem_key, new_conv)
    
    return backbone


def _infer_stage_out_channels(stage: nn.Module) -> int:
    """Infer stage output channels for ConvNeXt-like stages."""
    if hasattr(stage, 'blocks') and len(stage.blocks) > 0:
        block0 = stage.blocks[0]
        if hasattr(block0, 'conv_dw') and isinstance(block0.conv_dw, nn.Conv2d):
            return int(block0.conv_dw.weight.shape[0])
        if hasattr(block0, 'dwconv') and isinstance(block0.dwconv, nn.Conv2d):
            return int(block0.dwconv.weight.shape[0])

    if hasattr(stage, 'downsample'):
        for layer in stage.downsample.modules():
            if isinstance(layer, nn.Conv2d):
                return int(layer.out_channels)

    last_conv = None
    for layer in stage.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv = layer
    if last_conv is not None:
        return int(last_conv.out_channels)

    raise RuntimeError("Unable to infer stage output channels for CBAM injection")


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("SpatialAttention kernel_size must be odd")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class StageWithCBAM(nn.Module):
    """Wrap a stage and apply CBAM on its output."""
    def __init__(self, stage: nn.Module, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.stage = stage
        self.cbam = CBAM(channels=channels, reduction=reduction, spatial_kernel=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage(x)
        x = self.cbam(x)
        return x


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


def get_convnextv2_tiny(num_classes=17, pretrained=True):
    """
    获取 ConvNeXtV2 Tiny 模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: ConvNeXtV2 Tiny 模型
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm library is required for ConvNeXtV2 models. Install with: pip install timm")
    
    model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=pretrained, num_classes=0)
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes)
    
    return model


def get_convnextv2_base(num_classes=17, pretrained=True):
    """
    获取 ConvNeXtV2 Base 模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: ConvNeXtV2 Base 模型
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm library is required for ConvNeXtV2 models. Install with: pip install timm")
    
    model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=pretrained, num_classes=0)
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes)
    
    return model


def get_convnextv2_large(num_classes=17, pretrained=True):
    """
    获取 ConvNeXtV2 Large 模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: ConvNeXtV2 Large 模型
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm library is required for ConvNeXtV2 models. Install with: pip install timm")
    
    model = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k', pretrained=pretrained, num_classes=0)
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes)
    
    return model


def get_convnextv2_huge(num_classes=17, pretrained=True):
    """
    获取 ConvNeXtV2 Huge 模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: ConvNeXtV2 Huge 模型
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm library is required for ConvNeXtV2 models. Install with: pip install timm")
    
    model = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k', pretrained=pretrained, num_classes=0)
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes)
    
    return model


class ModelClassifier(nn.Module):
    """
    通用模型分类器包装类，支持 EfficientNetV2 和 ConvNeXt
    支持可变的输入通道数（3通道或5通道）
    """
    def __init__(
        self,
        num_classes=17,
        model_type='efficientnetv2_s',
        pretrained=True,
        dropout=0.2,
        in_channels=3,
        use_cbam_stage3=False,
        cbam_reduction=16,
        cbam_spatial_kernel=7,
    ):
        super(ModelClassifier, self).__init__()
        self.in_channels = in_channels
        self.model_type = model_type
        self.use_cbam_stage3 = use_cbam_stage3
        self.cbam_reduction = cbam_reduction
        self.cbam_spatial_kernel = cbam_spatial_kernel
        
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
            if 'v2' in model_type:
                # ConvNeXtV2 模型
                if not TIMM_AVAILABLE:
                    raise ImportError("timm library is required for ConvNeXtV2 models. Install with: pip install timm")
                
                sub_type = model_type.split('_')[-1]  # 'tiny', 'base', 'large', 'huge'
                timm_model_name = f'convnextv2_{sub_type}.fcmae_ft_in22k_in1k'
                
                # 首先尝试从本地加载权重
                local_weight_path = self._get_local_weight(timm_model_name, pretrained)
                
                if local_weight_path:
                    print(f"✓ 从本地加载权重: {local_weight_path}")
                    # 加载没有预训练的模型，然后手动加载权重
                    self.backbone = timm.create_model(timm_model_name, pretrained=False, num_classes=num_classes)
                    try:
                        # 支持 .safetensors 格式
                        if local_weight_path.endswith('.safetensors'):
                            if SAFETENSORS_AVAILABLE:
                                state_dict = load_file(local_weight_path)
                            else:
                                raise ImportError("safetensors library is required to load .safetensors files. Install with: pip install safetensors")
                        else:
                            state_dict = torch.load(local_weight_path, map_location='cpu', weights_only=False)
                        
                        # 跳过类别数不匹配的分类头权重
                        filtered_state_dict = {}
                        for k, v in state_dict.items():
                            # 跳过 head.fc 相关的权重（这些会因为类别数不同而不兼容）
                            if 'head.fc' not in k:
                                filtered_state_dict[k] = v
                            else:
                                print(f"  跳过权重: {k} (类别数不匹配)")
                        
                        self.backbone.load_state_dict(filtered_state_dict, strict=False)
                        print(f"✓ 权重加载成功")
                    except Exception as e:
                        print(f"⚠ 权重加载失败: {e}")
                        print(f"  使用随机初始化的模型")
                else:
                    # 本地权重不存在：若 pretrained=True，则尝试在线下载；失败后回退随机初始化
                    print(f"⚠ 本地权重不存在: {timm_model_name}")
                    if pretrained:
                        print("  尝试在线下载预训练权重...")
                        try:
                            self.backbone = timm.create_model(
                                timm_model_name,
                                pretrained=True,
                                num_classes=num_classes,
                            )
                            print("✓ 在线下载并加载预训练权重成功")
                        except Exception as e:
                            print(f"⚠ 在线下载失败: {e}")
                            print("  回退为随机初始化模型")
                            self.backbone = timm.create_model(
                                timm_model_name,
                                pretrained=False,
                                num_classes=num_classes,
                            )
                    else:
                        print("  使用随机初始化的模型")
                        self.backbone = timm.create_model(
                            timm_model_name,
                            pretrained=False,
                            num_classes=num_classes,
                        )
                
                # 适配输入通道数
                if in_channels != 3:
                    self.backbone = _adapt_stem_conv_in_channels(self.backbone, in_channels=in_channels, model_type='convnextv2')
                
                # 如果需要添加 dropout，直接修改分类头
                if hasattr(self.backbone, 'head'):
                    if isinstance(self.backbone.head, nn.Linear):
                        # 替换为带 dropout 的分类头
                        in_features = self.backbone.head.in_features
                        self.backbone.head = nn.Sequential(
                            nn.Dropout(p=dropout, inplace=True),
                            nn.Linear(in_features, num_classes)
                        )

                if self.use_cbam_stage3:
                    self._inject_convnextv2_stage3_cbam()
            else:
                # ConvNeXt V1 模型
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

    def _inject_convnextv2_stage3_cbam(self):
        """Inject CBAM only into ConvNeXtV2 stage3 (0-based index 2)."""
        if not hasattr(self.backbone, 'stages'):
            raise RuntimeError("Current ConvNeXtV2 backbone does not expose 'stages' for CBAM injection")

        stages = self.backbone.stages
        if len(stages) <= 2:
            raise RuntimeError(f"ConvNeXtV2 expects at least 3 stages, got {len(stages)}")

        stage3 = stages[2]
        out_channels = _infer_stage_out_channels(stage3)
        stages[2] = StageWithCBAM(
            stage=stage3,
            channels=out_channels,
            reduction=self.cbam_reduction,
            spatial_kernel=self.cbam_spatial_kernel,
        )
        print(f"✓ CBAM injected into ConvNeXtV2 stage3 with channels={out_channels}")
    
    def _get_local_weight(self, model_name, pretrained):
        """获取本地权重文件路径"""
        if not pretrained or not Config.PRETRAINED_WEIGHTS_DIR:
            return None
        
        weights_dir = Path(Config.PRETRAINED_WEIGHTS_DIR).absolute()
        
        # 尝试1: 直接查找与模型名称匹配的文件
        local_weight = weights_dir / f"{model_name}.safetensors"
        if local_weight.exists():
            return str(local_weight)
        
        # 尝试2: 查找timm格式的权重文件结构
        timm_pattern = str(weights_dir / f"hub/models--timm--{model_name}*/snapshots/*/model.safetensors")
        timm_weights = glob.glob(timm_pattern)
        
        if timm_weights:
            # 返回找到的第一个权重文件
            return timm_weights[0]
        
        # 尝试3: 查找包含模型名称的所有safetensors文件
        general_pattern = str(weights_dir / f"**/*{model_name}*.safetensors")
        general_weights = glob.glob(general_pattern, recursive=True)
        
        if general_weights:
            # 返回找到的第一个权重文件
            return general_weights[0]
        
        print(f"警告: 本地权重不存在: {model_name}")
        print(f"在路径 {weights_dir} 下搜索了以下模式:")
        print(f"1. {local_weight}")
        print(f"2. {timm_pattern}")
        print(f"3. {general_pattern}")
        return None

    def get_spatial_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """返回用于 CAM 的空间特征图 [B, C, H, W]。"""
        if self.model_type.startswith('efficientnetv2'):
            return self.backbone.features(x)

        if self.model_type.startswith('convnext'):
            if hasattr(self.backbone, 'forward_features'):
                feat = self.backbone.forward_features(x)
            elif hasattr(self.backbone, 'features'):
                feat = self.backbone.features(x)
            else:
                raise RuntimeError("Current backbone does not expose forward_features/features for OcCaMix")

            if isinstance(feat, (tuple, list)):
                feat = feat[0]

            if feat.ndim == 4 and feat.shape[-1] == self.get_classifier_weight().shape[1]:
                # 兼容 [B, H, W, C]
                feat = feat.permute(0, 3, 1, 2).contiguous()

            if feat.ndim != 4:
                raise RuntimeError(f"OcCaMix expects 4D feature map [B,C,H,W], got {tuple(feat.shape)}")

            return feat

        raise RuntimeError(f"Unsupported model type for OcCaMix: {self.model_type}")

    def get_classifier_weight(self) -> torch.Tensor:
        """返回最终分类线性层权重 [num_classes, C]。"""
        if self.model_type.startswith('efficientnetv2'):
            head = self.backbone.classifier
            linear = _find_last_linear(head)
            if linear is None:
                raise RuntimeError("No linear classifier found in EfficientNetV2 classifier")
            return linear.weight

        if self.model_type.startswith('convnext'):
            if hasattr(self.backbone, 'head'):
                linear = _find_last_linear(self.backbone.head)
                if linear is not None:
                    return linear.weight

            if hasattr(self.backbone, 'classifier'):
                linear = _find_last_linear(self.backbone.classifier)
                if linear is not None:
                    return linear.weight

            raise RuntimeError("No linear classifier found in ConvNeXt head/classifier")

        raise RuntimeError(f"Unsupported model type for classifier weight extraction: {self.model_type}")

    def get_feature_dim(self) -> int:
        weight = self.get_classifier_weight()
        if weight.ndim != 2:
            raise RuntimeError(f"Classifier weight should be 2D, got {tuple(weight.shape)}")
        return int(weight.shape[1])

    def extract_pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled pre-classifier features with shape [B, C]."""
        if self.model_type.startswith('efficientnetv2'):
            feat = self.backbone.features(x)
            feat = self.backbone.avgpool(feat)
            return torch.flatten(feat, 1)

        if self.model_type.startswith('convnext'):
            if hasattr(self.backbone, 'forward_features'):
                feat = self.backbone.forward_features(x)
            elif hasattr(self.backbone, 'features'):
                feat = self.backbone.features(x)
            else:
                raise RuntimeError("Current ConvNeXt backbone does not expose forward_features/features")

            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if feat.ndim == 2:
                return feat
            if feat.ndim != 4:
                raise RuntimeError(f"Expected 2D or 4D features, got {tuple(feat.shape)}")

            feature_dim = self.get_feature_dim()
            if feat.shape[1] == feature_dim:
                return feat.mean(dim=(-2, -1))
            if feat.shape[-1] == feature_dim:
                feat = feat.permute(0, 3, 1, 2).contiguous()
                return feat.mean(dim=(-2, -1))
            return feat.mean(dim=(-2, -1))

        raise RuntimeError(f"Unsupported model type for feature extraction: {self.model_type}")

    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """获取特征向量（用于特征提取或可视化）"""
        return self.extract_pooled_features(x)


class DualViewConvNeXtClassifier(nn.Module):
    """Dual-branch classifier for raw/detail views with fusion and auxiliary heads."""
    def __init__(
        self,
        num_classes=17,
        model_type='convnextv2_base',
        pretrained=True,
        dropout=0.2,
        fusion_hidden_dim=1024,
        fusion_dropout=0.2,
        use_cbam_stage3=False,
        cbam_reduction=16,
        cbam_spatial_kernel=7,
    ):
        super().__init__()
        self.raw_branch = ModelClassifier(
            num_classes=num_classes,
            model_type=model_type,
            pretrained=pretrained,
            dropout=dropout,
            in_channels=3,
            use_cbam_stage3=use_cbam_stage3,
            cbam_reduction=cbam_reduction,
            cbam_spatial_kernel=cbam_spatial_kernel,
        )
        self.detail_branch = ModelClassifier(
            num_classes=num_classes,
            model_type=model_type,
            pretrained=pretrained,
            dropout=dropout,
            in_channels=3,
            use_cbam_stage3=use_cbam_stage3,
            cbam_reduction=cbam_reduction,
            cbam_spatial_kernel=cbam_spatial_kernel,
        )

        feature_dim = self.raw_branch.get_feature_dim()
        hidden_dim = max(1, int(fusion_hidden_dim))
        self.raw_aux_head = nn.Linear(feature_dim, num_classes)
        self.detail_aux_head = nn.Linear(feature_dim, num_classes)
        self.fusion_head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=fusion_dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, raw_x: torch.Tensor, detail_x: torch.Tensor):
        raw_feat = self.raw_branch.extract_pooled_features(raw_x)
        detail_feat = self.detail_branch.extract_pooled_features(detail_x)
        fusion_feat = torch.cat([raw_feat, detail_feat], dim=1)
        return {
            'fusion_logits': self.fusion_head(fusion_feat),
            'raw_logits': self.raw_aux_head(raw_feat),
            'detail_logits': self.detail_aux_head(detail_feat),
        }


class SingleBackbone6ChClassifier(nn.Module):
    """Single-backbone classifier that consumes concatenated raw/detail 6-channel inputs."""
    def __init__(
        self,
        num_classes=17,
        model_type='convnextv2_base',
        pretrained=True,
        dropout=0.2,
        use_cbam_stage3=False,
        cbam_reduction=16,
        cbam_spatial_kernel=7,
    ):
        super().__init__()
        self.backbone = ModelClassifier(
            num_classes=num_classes,
            model_type=model_type,
            pretrained=pretrained,
            dropout=dropout,
            in_channels=6,
            use_cbam_stage3=use_cbam_stage3,
            cbam_reduction=cbam_reduction,
            cbam_spatial_kernel=cbam_spatial_kernel,
        )

    def forward(self, stacked_x: torch.Tensor):
        return {
            'fusion_logits': self.backbone(stacked_x),
        }


class SingleBackbone3ChClassifier(nn.Module):
    """Single-backbone classifier for train2 single-view 3-channel experiments."""
    def __init__(
        self,
        num_classes=17,
        model_type='convnextv2_base',
        pretrained=True,
        dropout=0.2,
        use_cbam_stage3=False,
        cbam_reduction=16,
        cbam_spatial_kernel=7,
    ):
        super().__init__()
        self.backbone = ModelClassifier(
            num_classes=num_classes,
            model_type=model_type,
            pretrained=pretrained,
            dropout=dropout,
            in_channels=3,
            use_cbam_stage3=use_cbam_stage3,
            cbam_reduction=cbam_reduction,
            cbam_spatial_kernel=cbam_spatial_kernel,
        )

    def forward(self, x: torch.Tensor):
        return {
            'fusion_logits': self.backbone(x),
        }

    def get_spatial_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_spatial_feature_map(x)

    def get_classifier_weight(self) -> torch.Tensor:
        return self.backbone.get_classifier_weight()

    def extract_pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.extract_pooled_features(x)


def build_model(config, pretrained=True):
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
        pretrained=pretrained,
        dropout=0.2,
        in_channels=in_channels,
        use_cbam_stage3=getattr(config, 'USE_CBAM_STAGE3', False),
        cbam_reduction=getattr(config, 'CBAM_REDUCTION', 16),
        cbam_spatial_kernel=getattr(config, 'CBAM_SPATIAL_KERNEL', 7),
    )
    return model


def build_dualview_model(config, pretrained=True):
    if getattr(config, 'USE_FREQ_CHANNELS', False):
        raise ValueError("Dual-view model does not support USE_FREQ_CHANNELS=True in this version.")

    if config.PRETRAINED_WEIGHTS_DIR is not None:
        os.makedirs(config.PRETRAINED_WEIGHTS_DIR, exist_ok=True)
        torch.hub.set_dir(config.PRETRAINED_WEIGHTS_DIR)

    model_variant = getattr(config, 'MODEL_VARIANT', 'dual_branch')
    common_kwargs = {
        'num_classes': config.NUM_CLASSES,
        'model_type': config.MODEL_TYPE,
        'pretrained': pretrained,
        'dropout': getattr(config, 'DROPOUT', 0.2),
        'use_cbam_stage3': getattr(config, 'USE_CBAM_STAGE3', False),
        'cbam_reduction': getattr(config, 'CBAM_REDUCTION', 16),
        'cbam_spatial_kernel': getattr(config, 'CBAM_SPATIAL_KERNEL', 7),
    }

    if model_variant == 'dual_branch':
        model = DualViewConvNeXtClassifier(
            fusion_hidden_dim=getattr(config, 'FUSION_HIDDEN_DIM', 1024),
            fusion_dropout=getattr(config, 'FUSION_DROPOUT', 0.2),
            **common_kwargs,
        )
    elif model_variant == 'single_backbone_6ch':
        model = SingleBackbone6ChClassifier(**common_kwargs)
    else:
        raise ValueError(f"Unknown MODEL_VARIANT: {model_variant}")
    return model


def build_train2_model(config, pretrained=True):
    train_mode = getattr(config, 'TRAIN_MODE', 'dual_view')
    if train_mode == 'dual_view':
        return build_dualview_model(config, pretrained=pretrained)
    if train_mode == 'single_view':
        if getattr(config, 'USE_FREQ_CHANNELS', False):
            raise ValueError("train2 single-view mode does not support USE_FREQ_CHANNELS=True.")
        if config.PRETRAINED_WEIGHTS_DIR is not None:
            os.makedirs(config.PRETRAINED_WEIGHTS_DIR, exist_ok=True)
            torch.hub.set_dir(config.PRETRAINED_WEIGHTS_DIR)
        return SingleBackbone3ChClassifier(
            num_classes=config.NUM_CLASSES,
            model_type=config.MODEL_TYPE,
            pretrained=pretrained,
            dropout=getattr(config, 'DROPOUT', 0.2),
            use_cbam_stage3=getattr(config, 'USE_CBAM_STAGE3', False),
            cbam_reduction=getattr(config, 'CBAM_REDUCTION', 16),
            cbam_spatial_kernel=getattr(config, 'CBAM_SPATIAL_KERNEL', 7),
        )
    raise ValueError(f"Unknown TRAIN_MODE: {train_mode}")
