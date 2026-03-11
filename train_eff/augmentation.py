import torch
import numpy as np


def mixup_data(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4):
    """
    MixUp 数据增强
    
    Args:
        images: [B, C, H, W]
        labels: [B] 类别索引
        alpha:  Beta分布参数，越大混合越均匀，推荐 0.2~0.8
    
    Returns:
        mixed_images: [B, C, H, W]
        labels_a:     原始标签
        labels_b:     混合标签
        lam:          混合比例（标量）
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    B = images.size(0)
    rand_idx = torch.randperm(B, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[rand_idx]
    labels_a = labels
    labels_b = labels[rand_idx]

    return mixed_images, labels_a, labels_b, lam


def cutmix_data(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    """
    CutMix 数据增强
    
    Args:
        images: [B, C, H, W]
        labels: [B] 类别索引
        alpha:  Beta分布参数，推荐 0.5~1.5
    
    Returns:
        mixed_images: [B, C, H, W]
        labels_a:     原始标签
        labels_b:     混合标签
        lam:          原始图像保留的面积比例
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    B, C, H, W = images.shape
    rand_idx = torch.randperm(B, device=images.device)

    # 生成随机裁剪框
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]

    # 用实际面积修正 lam
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)

    labels_a = labels
    labels_b = labels[rand_idx]

    return mixed_images, labels_a, labels_b, lam


def mixed_criterion(criterion, pred, labels_a, labels_b, lam):
    """
    混合损失计算：lam * loss(pred, a) + (1-lam) * loss(pred, b)
    
    注意：criterion 需要支持逐样本输出（reduction='none'）
          或者直接用两次 mean 加权
    """
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)