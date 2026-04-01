import os
import math
import random
import torch
from torchvision.utils import save_image
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
from loss import FocalLoss
from config import Config
from dataset import get_dataloaders
from model import build_model
from augmentation import mixup_data, cutmix_data, cutmix_data_yolo, occamix_data, mixed_criterion
from yolo_cutmix import YoloCutMixHelper
from utils import (
    set_seed, save_checkpoint, load_checkpoint,
    AverageMeter, accuracy, log_training, get_current_time,
    save_error_samples
)
from metrics import (
    compute_class_metrics, print_class_metrics,
    print_confusion_matrix
)


def save_aug_preview_batch(
    images_orig: torch.Tensor,
    images_aug: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam,
    epoch: int,
    batch_idx: int,
    aug_type: str,
    save_root: str,
    max_samples: int = 4,
):
    """
    保存一个batch中的若干“原图/增强图”对照图。
    输出图每张包含2列：左原图，右增强图。
    """
    out_dir = Path(save_root) / "aug_preview" / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bsz = images_orig.size(0)
    n = min(max_samples, bsz)

    # lam 可能是标量（cutmix/mixup）或向量（occamix）
    lam_is_tensor = torch.is_tensor(lam)

    for i in range(n):
        lam_i = float(lam[i].item()) if lam_is_tensor else float(lam)
        la = int(labels_a[i].item())
        lb = int(labels_b[i].item())

        pair = torch.stack(
            [images_orig[i].detach().cpu(), images_aug[i].detach().cpu()], dim=0
        )  # [2, C, H, W]

        fname = (
            f"{aug_type}_e{epoch:03d}_b{batch_idx:04d}_i{i:02d}"
            f"_la{la}_lb{lb}_lam{lam_i:.3f}.png"
        )
        save_path = out_dir / fname

        save_image(
            pair,
            str(save_path),
            nrow=2,            # 左原图，右增强图
            normalize=True,    # 便于直接查看
        )


def apply_post_rgb_augment(images: torch.Tensor, config, yolo_area_ratios=None):
    """在 batch 混合增强之后再做 RGB 几何/颜色增强。"""
    hflip_p = float(getattr(config, 'POST_AUG_HFLIP_P', 0.5))
    rotate_deg = float(getattr(config, 'POST_AUG_ROTATE_DEGREES', 15))
    brightness = float(getattr(config, 'POST_AUG_BRIGHTNESS', 0.2))
    contrast = float(getattr(config, 'POST_AUG_CONTRAST', 0.2))
    saturation = float(getattr(config, 'POST_AUG_SATURATION', 0.2))
    hue = float(getattr(config, 'POST_AUG_HUE', 0.1))
    scale_area_adaptive = bool(getattr(config, 'POST_AUG_SCALE_AREA_ADAPTIVE', False))
    scale_area_prob = float(np.clip(getattr(config, 'POST_AUG_SCALE_AREA_PROB', 0.5), 0.0, 1.0))
    scale_area_small_thres = float(getattr(config, 'POST_AUG_SCALE_AREA_SMALL_THRES', 0.04))
    scale_area_large_thres = float(getattr(config, 'POST_AUG_SCALE_AREA_LARGE_THRES', 0.10))
    scale_jitter_small_box = float(getattr(config, 'POST_AUG_SCALE_JITTER_SMALL_BOX', 0.05))
    scale_jitter_mid_box = float(getattr(config, 'POST_AUG_SCALE_JITTER_MID_BOX', 0.05))
    scale_jitter_large_box = float(getattr(config, 'POST_AUG_SCALE_JITTER_LARGE_BOX', 0.03))
    noise_std = float(getattr(config, 'POST_AUG_NOISE_STD', 0.015))
    sp_noise_p = float(getattr(config, 'POST_AUG_SP_NOISE_P', 0.003))

    if scale_area_large_thres < scale_area_small_thres:
        scale_area_large_thres = scale_area_small_thres

    scale_stats = {
        'post_scale_small': 0,
        'post_scale_mid': 0,
        'post_scale_large': 0,
        'post_scale_fallback': 0,
        'post_scale_not_triggered': 0,
    }

    out = []
    for i in range(images.size(0)):
        x = images[i]
        _, h, w = x.shape

        # 自适应缩放：由面积分段控制区间，并按概率触发
        if scale_area_adaptive and (random.random() < scale_area_prob):
            scale_low = 1.0
            scale_high = 1.0

            area_ratio = None
            if yolo_area_ratios is not None and i < len(yolo_area_ratios):
                candidate = yolo_area_ratios[i]
                if candidate is not None:
                    area_ratio = float(candidate)

            if area_ratio is None or area_ratio <= 0:
                mj = max(0.0, scale_jitter_mid_box)
                scale_low = max(0.1, 1.0 - mj)
                scale_high = 1.0 + mj
                scale_stats['post_scale_fallback'] += 1
            elif area_ratio < scale_area_small_thres:
                sj = max(0.0, scale_jitter_small_box)
                scale_low = 1.0
                scale_high = 1.0 + sj
                scale_stats['post_scale_small'] += 1
            elif area_ratio > scale_area_large_thres:
                lj = max(0.0, scale_jitter_large_box)
                scale_low = max(0.1, 1.0 - lj)
                scale_high = 1.0
                scale_stats['post_scale_large'] += 1
            else:
                mj = max(0.0, scale_jitter_mid_box)
                scale_low = max(0.1, 1.0 - mj)
                scale_high = 1.0 + mj
                scale_stats['post_scale_mid'] += 1

            if scale_high < scale_low:
                scale_high = scale_low
            if scale_high > scale_low:
                scale = random.uniform(scale_low, scale_high)
                nh = max(1, int(round(h * scale)))
                nw = max(1, int(round(w * scale)))
                x_scaled = TF.resize(x, [nh, nw], interpolation=InterpolationMode.BILINEAR)
                if nh >= h and nw >= w:
                    top = (nh - h) // 2
                    left = (nw - w) // 2
                    x = TF.crop(x_scaled, top, left, h, w)
                else:
                    pad_t = (h - nh) // 2
                    pad_b = h - nh - pad_t
                    pad_l = (w - nw) // 2
                    pad_r = w - nw - pad_l
                    x = TF.pad(x_scaled, [pad_l, pad_t, pad_r, pad_b], fill=0.0)
        else:
            scale_stats['post_scale_not_triggered'] += 1

        if random.random() < hflip_p:
            x = TF.hflip(x)

        if rotate_deg > 0:
            angle = random.uniform(-rotate_deg, rotate_deg)
            x = TF.rotate(x, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)

        if brightness > 0:
            x = TF.adjust_brightness(x, 1.0 + random.uniform(-brightness, brightness))
        if contrast > 0:
            x = TF.adjust_contrast(x, 1.0 + random.uniform(-contrast, contrast))
        if saturation > 0:
            x = TF.adjust_saturation(x, 1.0 + random.uniform(-saturation, saturation))
        if hue > 0:
            x = TF.adjust_hue(x, random.uniform(-hue, hue))

        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        if sp_noise_p > 0:
            p = max(0.0, min(1.0, sp_noise_p))
            rnd = torch.rand((h, w), device=x.device)
            salt = rnd < (0.5 * p)
            pepper = (rnd >= (0.5 * p)) & (rnd < p)
            if salt.any():
                x[:, salt] = 1.0
            if pepper.any():
                x[:, pepper] = 0.0

        out.append(torch.clamp(x, 0.0, 1.0))
    return torch.stack(out, dim=0), scale_stats


def normalize_rgb_batch(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config, yolo_cutmix_helper=None):
    """训练一个 epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    

    aug_type = getattr(config, 'AUG_TYPE', 'none')
    aug_alpha = getattr(config, 'AUG_ALPHA', 1.0)
    aug_prob = getattr(config, 'AUG_PROB', 0.5)
    occamix_n = getattr(config, 'OCCAMIX_N', 6)
    occamix_seg_min = getattr(config, 'OCCAMIX_SEG_MIN', 20)
    occamix_seg_max = getattr(config, 'OCCAMIX_SEG_MAX', 50)
    occamix_compactness = getattr(config, 'OCCAMIX_COMPACTNESS', 10.0)
    
    preview_enable = getattr(config, "SAVE_AUG_PREVIEW", True)
    preview_max_batches = getattr(config, "SAVE_AUG_PREVIEW_MAX_BATCHES", 2)
    preview_max_samples = getattr(config, "SAVE_AUG_PREVIEW_MAX_SAMPLES", 4)
    preview_saved_batches = 0

    yolo_stats_epoch = {
        'total': 0,
        'applied': 0,
        'skipped': 0,
        'skipped_missing': 0,
        'skipped_empty': 0,
        'skipped_invalid': 0,
        'skipped_too_small': 0,
        'skipped_too_large': 0,
        'skipped_dst_invalid': 0,
        'pair_random': 0,
        'pair_threshold_matched': 0,
        'pair_threshold_fallback_random': 0,
        'pair_no_area_fallback': 0,
        'pair_ratio_min_current': 0.0,
        'pair_ratio_max_current': 0.0,
        'route_to_mixup': 0,
        'route_to_none': 0,
        'post_scale_small': 0,
        'post_scale_mid': 0,
        'post_scale_large': 0,
        'post_scale_fallback': 0,
        'post_scale_not_triggered': 0,
    }

    sample_routing_enable = getattr(config, 'YOLO_CUTMIX_SAMPLE_ROUTING_ENABLE', False)
    non_eligible_policy = getattr(config, 'YOLO_CUTMIX_NON_ELIGIBLE_POLICY', 'none')
    non_eligible_mixup_alpha = getattr(config, 'YOLO_CUTMIX_NON_ELIGIBLE_MIXUP_ALPHA', 0.4)
    post_aug_enable = bool(getattr(config, 'POST_AUG_ENABLE', True))
    use_freq_channels = bool(getattr(config, 'USE_FREQ_CHANNELS', False))

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels, img_paths) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images_orig = images.clone()  # 用于可视化对照
        labels_a, labels_b, lam = labels, labels, 1.0
        aug_applied = False
        yolo_area_ratios_batch = None

        # ====== 你的增强分支（示例）======
        # if aug_type == 'cutmix' and torch.rand(1).item() < aug_prob:
        #     images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=aug_alpha)
        #     aug_applied = True
        #
        # elif aug_type == 'occamix' and torch.rand(1).item() < aug_prob:
        #     images, labels_a, labels_b, lam = occamix_data(
        #         images, labels, model,
        #         n_top=occamix_n,
        #         n_seg_max=occamix_seg_max,
        #         n_seg_min=occamix_seg_min,
        #         compactness=occamix_compactness,
        #     )
        #     aug_applied = True
        # ================================

        # 增强
        use_aug = (aug_type != 'none') and (np.random.rand() < aug_prob)
        chosen = 'none'
        if use_aug:
            if aug_type == 'both':
                # 随机选一种
                chosen = np.random.choice(['mixup', 'cutmix'])
            elif aug_type == 'both_all':
                # 随机选一种（包含 occamix / cutmix_yolo）
                candidates = ['mixup', 'cutmix', 'occamix']
                if yolo_cutmix_helper is not None:
                    candidates.append('cutmix_yolo')
                chosen = np.random.choice(candidates)
            else:
                chosen = aug_type

            if chosen == 'mixup':
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=aug_alpha)
            elif chosen == 'cutmix':
                images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=aug_alpha)
            elif chosen == 'occamix':
                images, labels_a, labels_b, lam = occamix_data(
                    images,
                    labels,
                    model,
                    n_top=occamix_n,
                    n_seg_max=occamix_seg_max,
                    n_seg_min=occamix_seg_min,
                    compactness=occamix_compactness,
                )
            elif chosen == 'cutmix_yolo':
                images, labels_a, labels_b, lam, batch_stats, applied_mask, rand_idx = cutmix_data_yolo(
                    images,
                    labels,
                    img_paths,
                    yolo_cutmix_helper,
                    alpha=aug_alpha,
                )
                yolo_stats_epoch['total'] += int(batch_stats.get('total', 0))
                yolo_stats_epoch['applied'] += int(batch_stats.get('applied', 0))
                yolo_stats_epoch['skipped'] += int(batch_stats.get('skipped', 0))
                yolo_stats_epoch['skipped_missing'] += int(batch_stats.get('skipped_missing', 0))
                yolo_stats_epoch['skipped_empty'] += int(batch_stats.get('skipped_empty', 0))
                yolo_stats_epoch['skipped_invalid'] += int(batch_stats.get('skipped_invalid', 0))
                yolo_stats_epoch['skipped_too_small'] += int(batch_stats.get('skipped_too_small', 0))
                yolo_stats_epoch['skipped_too_large'] += int(batch_stats.get('skipped_too_large', 0))
                yolo_stats_epoch['skipped_dst_invalid'] += int(batch_stats.get('skipped_dst_invalid', 0))
                yolo_stats_epoch['pair_random'] += int(batch_stats.get('pair_random', 0))
                yolo_stats_epoch['pair_threshold_matched'] += int(batch_stats.get('pair_threshold_matched', 0))
                yolo_stats_epoch['pair_threshold_fallback_random'] += int(batch_stats.get('pair_threshold_fallback_random', 0))
                yolo_stats_epoch['pair_no_area_fallback'] += int(batch_stats.get('pair_no_area_fallback', 0))
                yolo_stats_epoch['pair_ratio_min_current'] = float(batch_stats.get('pair_ratio_min_current', yolo_stats_epoch['pair_ratio_min_current']))
                yolo_stats_epoch['pair_ratio_max_current'] = float(batch_stats.get('pair_ratio_max_current', yolo_stats_epoch['pair_ratio_max_current']))
                yolo_area_ratios_batch = batch_stats.get('area_ratios_min')
                if yolo_area_ratios_batch is None:
                    yolo_area_ratios_batch = batch_stats.get('area_ratios')

                if sample_routing_enable:
                    non_eligible_mask = ~applied_mask
                    non_eligible_count = int(non_eligible_mask.sum().item())
                    if non_eligible_count > 0:
                        if non_eligible_policy == 'mixup':
                            if non_eligible_mixup_alpha > 0:
                                lam_non = float(np.random.beta(non_eligible_mixup_alpha, non_eligible_mixup_alpha))
                            else:
                                lam_non = 1.0
                            # 仅对不适配样本执行 MixUp，保持同 batch 其余样本不变
                            images[non_eligible_mask] = (
                                lam_non * images_orig[non_eligible_mask]
                                + (1.0 - lam_non) * images_orig[rand_idx[non_eligible_mask]]
                            )
                            lam[non_eligible_mask] = lam_non
                            yolo_stats_epoch['route_to_mixup'] += non_eligible_count
                        else:
                            # 'none' 策略：保留原图，不做增强
                            images[non_eligible_mask] = images_orig[non_eligible_mask]
                            lam[non_eligible_mask] = 1.0
                            yolo_stats_epoch['route_to_none'] += non_eligible_count
            else:
                raise ValueError(f"Unknown augmentation type: {chosen}")

            aug_applied = True

        # 仅 RGB 训练：在混合增强后执行几何/颜色增强，并在入模前归一化
        if not use_freq_channels and images.size(1) == 3:
            if post_aug_enable:
                images, post_scale_stats = apply_post_rgb_augment(images, config, yolo_area_ratios=yolo_area_ratios_batch)
                yolo_stats_epoch['post_scale_small'] += int(post_scale_stats.get('post_scale_small', 0))
                yolo_stats_epoch['post_scale_mid'] += int(post_scale_stats.get('post_scale_mid', 0))
                yolo_stats_epoch['post_scale_large'] += int(post_scale_stats.get('post_scale_large', 0))
                yolo_stats_epoch['post_scale_fallback'] += int(post_scale_stats.get('post_scale_fallback', 0))
                yolo_stats_epoch['post_scale_not_triggered'] += int(post_scale_stats.get('post_scale_not_triggered', 0))
            images = normalize_rgb_batch(images)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)

        # 计算损失
        if isinstance(lam, torch.Tensor) and config.LOSS_TYPE == 'cross_entropy':
            lam_batch = lam.to(device=outputs.device, dtype=outputs.dtype).view(-1)
            loss_a = F.cross_entropy(
                outputs,
                labels_a,
                label_smoothing=config.LABEL_SMOOTHING,
                reduction='none'
            )
            loss_b = F.cross_entropy(
                outputs,
                labels_b,
                label_smoothing=config.LABEL_SMOOTHING,
                reduction='none'
            )
            loss = (lam_batch * loss_a + (1.0 - lam_batch) * loss_b).mean()
        elif isinstance(lam, torch.Tensor):
            lam_batch = lam.to(device=outputs.device, dtype=outputs.dtype).view(-1)
            if hasattr(criterion, 'reduction'):
                old_reduction = criterion.reduction
                try:
                    criterion.reduction = 'none'
                    loss_a = criterion(outputs, labels_a)
                    loss_b = criterion(outputs, labels_b)
                finally:
                    criterion.reduction = old_reduction
                loss = (lam_batch * loss_a + (1.0 - lam_batch) * loss_b).mean()
            else:
                # 回退：若不支持逐样本，使用 batch 平均比例
                lam_scalar = float(lam_batch.mean().item())
                loss = mixed_criterion(criterion, outputs, labels_a, labels_b, lam_scalar)
        else:
            loss = mixed_criterion(criterion, outputs, labels_a, labels_b, lam)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        acc1 = accuracy(outputs, labels, topk=(1,))[0]
        
        # 更新统计
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{top1.avg:.2f}%'
        })
        
        # 仅保存前1~2个发生增强的batch
        if (
            preview_enable
            and aug_applied
            and preview_saved_batches < preview_max_batches
        ):
            save_aug_preview_batch(
                images_orig=images_orig,
                images_aug=images,
                labels_a=labels_a,
                labels_b=labels_b,
                lam=lam,
                epoch=epoch,
                batch_idx=batch_idx,
                aug_type=aug_type,
                save_root=config.CHECKPOINT_DIR,
                max_samples=preview_max_samples,
            )
            preview_saved_batches += 1

    total = max(1, yolo_stats_epoch['total'])
    yolo_stats_epoch['applied_ratio'] = yolo_stats_epoch['applied'] / total
    yolo_stats_epoch['skipped_ratio'] = yolo_stats_epoch['skipped'] / total
    yolo_stats_epoch['pair_random_ratio'] = yolo_stats_epoch['pair_random'] / total
    yolo_stats_epoch['pair_threshold_matched_ratio'] = yolo_stats_epoch['pair_threshold_matched'] / total
    yolo_stats_epoch['pair_threshold_fallback_random_ratio'] = yolo_stats_epoch['pair_threshold_fallback_random'] / total
    yolo_stats_epoch['pair_no_area_fallback_ratio'] = yolo_stats_epoch['pair_no_area_fallback'] / total
    yolo_stats_epoch['route_to_mixup_ratio'] = yolo_stats_epoch['route_to_mixup'] / total
    yolo_stats_epoch['route_to_none_ratio'] = yolo_stats_epoch['route_to_none'] / total
    post_scale_total = max(
        1,
        yolo_stats_epoch['post_scale_small']
        + yolo_stats_epoch['post_scale_mid']
        + yolo_stats_epoch['post_scale_large']
        + yolo_stats_epoch['post_scale_fallback']
        + yolo_stats_epoch['post_scale_not_triggered']
    )
    yolo_stats_epoch['post_scale_small_ratio'] = yolo_stats_epoch['post_scale_small'] / post_scale_total
    yolo_stats_epoch['post_scale_mid_ratio'] = yolo_stats_epoch['post_scale_mid'] / post_scale_total
    yolo_stats_epoch['post_scale_large_ratio'] = yolo_stats_epoch['post_scale_large'] / post_scale_total
    yolo_stats_epoch['post_scale_fallback_ratio'] = yolo_stats_epoch['post_scale_fallback'] / post_scale_total
    yolo_stats_epoch['post_scale_not_triggered_ratio'] = yolo_stats_epoch['post_scale_not_triggered'] / post_scale_total
    return losses.avg, top1.avg, yolo_stats_epoch


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    all_preds = []
    all_labels = []
    error_samples = []  # 记录错误样本
    
    pbar = tqdm(val_loader, desc="[Validate]")
    for images, labels, img_paths in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 计算准确率
        acc1 = accuracy(outputs, labels, topk=(1,))[0]
        
        # 更新统计
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        # 保存预测结果
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 记录错误样本
        probs = torch.softmax(outputs, dim=1)
        max_probs = probs.max(dim=1)
        
        for pred, label, img_path, max_prob in zip(preds.cpu().numpy(), labels.cpu().numpy(), img_paths, max_probs.values.cpu().numpy()):
            if pred != label:
                error_samples.append({
                    'img_path': img_path,
                    'pred_label': int(pred),
                    'true_label': int(label),
                    'confidence': float(max_prob)
                })
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{top1.avg:.2f}%'
        })
    
    return losses.avg, top1.avg, all_preds, all_labels, error_samples


def main():
    # 加载配置
    config = Config()
    
    # 设置随机种子
    set_seed(config.SEED)
    
    # 设置设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建日志文件
    log_file = os.path.join(config.CHECKPOINT_DIR, f"train_{get_current_time()}.log")
    
    # 加载数据
    print("Loading data...")
    train_loader, val_loader, class_names = get_dataloaders(config)
    print(f"Number of classes: {len(class_names)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Val samples: {len(val_loader.dataset)}")
    
    # 构建模型
    print("Building model...")
    print(f'model: {config.MODEL_TYPE}')
    print(f'Use frequency channels: {config.USE_FREQ_CHANNELS}')
    model = build_model(config)
    model = model.to(device)

    # 数据增强配置
    print(f"Data augmentation type: {config.AUG_TYPE}")
    print(f"Augmentation probability: {config.AUG_PROB}")
    if config.AUG_TYPE == 'occamix':
        print(f"OcCaMix N: {config.OCCAMIX_N}")
        print(f"OcCaMix seg min: {config.OCCAMIX_SEG_MIN}")
        print(f"OcCaMix seg max: {config.OCCAMIX_SEG_MAX}")
        print(f"OcCaMix compactness: {config.OCCAMIX_COMPACTNESS}")

    yolo_cutmix_helper = None
    if config.AUG_TYPE in ['cutmix_yolo', 'both_all'] and getattr(config, 'YOLO_CUTMIX_ENABLE', False):
        yolo_cutmix_helper = YoloCutMixHelper(
            cache_path=config.YOLO_CUTMIX_CACHE_PATH,
            train_dir=config.TRAIN_DIR,
            key_mode=config.YOLO_CUTMIX_KEY_MODE,
            fallback_mode=config.YOLO_CUTMIX_FALLBACK,
            min_box_area_ratio=config.YOLO_CUTMIX_MIN_BOX_AREA_RATIO,
            max_box_area_ratio=config.YOLO_CUTMIX_MAX_BOX_AREA_RATIO,
            center_tolerance_ratio=getattr(config, 'YOLO_CUTMIX_CENTER_TOLERANCE_RATIO', 0.10),
            debug_log=config.YOLO_CUTMIX_DEBUG_LOG,
            pair_random_prob=getattr(config, 'YOLO_CUTMIX_PAIR_RANDOM_PROB', 0.20),
            pair_area_ratio_min=getattr(config, 'YOLO_CUTMIX_PAIR_AREA_RATIO_MIN', 0.20),
            pair_area_ratio_max=getattr(config, 'YOLO_CUTMIX_PAIR_AREA_RATIO_MAX', 5.00),
            pair_area_ratio_min_target=getattr(config, 'YOLO_CUTMIX_PAIR_AREA_RATIO_MIN_TARGET', 0.30),
            pair_area_ratio_max_target=getattr(config, 'YOLO_CUTMIX_PAIR_AREA_RATIO_MAX_TARGET', 3.00),
            pair_schedule_start_ratio=getattr(config, 'YOLO_CUTMIX_PAIR_SCHEDULE_START_RATIO', 0.30),
            pair_schedule_end_ratio=getattr(config, 'YOLO_CUTMIX_PAIR_SCHEDULE_END_RATIO', 0.85),
        )

    
    # 多GPU配置
    if getattr(config, 'USE_MULTI_GPU', False) and torch.cuda.device_count() > 1:
        gpu_ids = getattr(config, 'GPU_IDS', list(range(torch.cuda.device_count())))
        print(f"Using DataParallel with GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    else:
        print(f"Using single GPU/device: {device}")
    
    if config.LOSS_TYPE == 'focal':
        # 可选：按类别样本数计算 alpha（类别越少，权重越大）
        alpha = None
        if getattr(config, 'USE_CLASS_ALPHA', False):
            # train_loader.dataset 需要有 targets 属性
            targets = torch.tensor(train_loader.dataset.labels)
            class_counts = torch.bincount(targets, minlength=config.NUM_CLASSES).float()
            alpha = 1.0 / (class_counts + 1e-6)
            alpha = alpha / alpha.sum() * config.NUM_CLASSES  # 归一化

        criterion = FocalLoss(
            gamma=config.FOCAL_GAMMA,
            alpha=alpha,
            label_smoothing=config.LABEL_SMOOTHING,
        )
        print(f"Using Focal Loss (gamma={config.FOCAL_GAMMA}, label_smoothing={config.LABEL_SMOOTHING})")
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=config.LABEL_SMOOTHING
        )
        print("Using CrossEntropy Loss")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS
    )
    
    # 训练参数
    best_acc = 0.0
    best_macro_f1 = 0.0
    best_val_loss = float('inf')
    start_epoch = 0
    
    # 检查是否有检查点可以恢复
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'latest.pth')
    if config.RESUME and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, best_acc, best_macro_f1, best_val_loss = load_checkpoint(
            model, optimizer, checkpoint_path, device
        )
        start_epoch += 1
    elif os.path.exists(checkpoint_path):
        print(f"Checkpoint found but RESUME is disabled. Starting fresh training.")
        print(f"To resume training, set RESUME = True in config.py")
    
    # 训练循环
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        if yolo_cutmix_helper is not None:
            progress = epoch / max(1, config.NUM_EPOCHS - 1)
            yolo_cutmix_helper.set_pair_schedule_progress(progress)
        
        # 训练
        train_loss, train_acc, yolo_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1, config,
            yolo_cutmix_helper=yolo_cutmix_helper,
        )

        if yolo_stats['total'] > 0:
            print(
                "YOLO-CutMix stats - "
                f"applied: {yolo_stats['applied_ratio']:.3f}, "
                f"skipped: {yolo_stats['skipped_ratio']:.3f}, "
                f"miss/empty/invalid: "
                f"{yolo_stats['skipped_missing']}/"
                f"{yolo_stats['skipped_empty']}/"
                f"{yolo_stats['skipped_invalid']}, "
                f"too_small/too_large: "
                f"{yolo_stats['skipped_too_small']}/"
                f"{yolo_stats['skipped_too_large']}, "
                f"dst_invalid: {yolo_stats['skipped_dst_invalid']}, "
                f"pair(rand/thres/fallback/no_area): "
                f"{yolo_stats['pair_random_ratio']:.3f}/"
                f"{yolo_stats['pair_threshold_matched_ratio']:.3f}/"
                f"{yolo_stats['pair_threshold_fallback_random_ratio']:.3f}/"
                f"{yolo_stats['pair_no_area_fallback_ratio']:.3f}, "
                f"pair_window[min,max]: "
                f"{yolo_stats['pair_ratio_min_current']:.3f}/"
                f"{yolo_stats['pair_ratio_max_current']:.3f}, "
                f"post_scale(small/mid/large/fallback): "
                f"{yolo_stats['post_scale_small_ratio']:.3f}/"
                f"{yolo_stats['post_scale_mid_ratio']:.3f}/"
                f"{yolo_stats['post_scale_large_ratio']:.3f}/"
                f"{yolo_stats['post_scale_fallback_ratio']:.3f}, "
                f"post_scale_not_triggered: {yolo_stats['post_scale_not_triggered_ratio']:.3f}, "
                f"route(mixup/none): "
                f"{yolo_stats['route_to_mixup_ratio']:.3f}/"
                f"{yolo_stats['route_to_none_ratio']:.3f}"
            )
        
        # 验证
        val_loss, val_acc = None, None
        val_macro_f1 = None
        if val_loader:
            val_loss, val_acc, all_preds, all_labels, error_samples = validate(model, val_loader, criterion, device)
            
            # 计算并打印类别级指标
            metrics_dict = compute_class_metrics(
                all_preds, all_labels, 
                num_classes=config.NUM_CLASSES, 
                class_names=class_names
            )
            print_class_metrics(metrics_dict, class_names)
            print_confusion_matrix(metrics_dict['confusion_matrix'], class_names)
            
            # 计算 macro-F1
            val_macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            # 保存错误样本图片
            if getattr(config, 'SAVE_ERROR_SAMPLES', False) and error_samples:
                error_samples_dir = getattr(config, 'ERROR_SAMPLES_DIR', 'eff/error_samples')
                save_error_samples(epoch+1, error_samples, error_samples_dir, class_names)
        
        # 更新学习率
        scheduler.step()
        
        # 记录日志
        log_training(
            epoch+1, config.NUM_EPOCHS,
            train_loss, train_acc,
            val_loss, val_acc,
            log_file
        )
        
        # 保存检查点
        is_best_acc = val_acc is not None and val_acc > best_acc
        is_best_macro_f1 = val_macro_f1 is not None and val_macro_f1 > best_macro_f1
        is_best_loss = val_loss is not None and val_loss < best_val_loss
        
        if is_best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, best_acc, best_macro_f1, best_val_loss,
                os.path.join(config.CHECKPOINT_DIR, 'best_acc.pth')
            )
            print(f"✓ Best accuracy updated: {best_acc:.2f}%")
        
        if is_best_macro_f1:
            best_macro_f1 = val_macro_f1
            save_checkpoint(
                model, optimizer, epoch, best_acc, best_macro_f1, best_val_loss,
                os.path.join(config.CHECKPOINT_DIR, 'best_macro_f1.pth')
            )
            print(f"✓ Best macro-F1 updated: {best_macro_f1:.4f}")

        if is_best_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, best_acc, best_macro_f1, best_val_loss,
                os.path.join(config.CHECKPOINT_DIR, 'best_loss.pth')
            )
            print(f"✓ Best val loss updated: {best_val_loss:.4f}")
        
        # 保存最新检查点
        save_checkpoint(
            model, optimizer, epoch, best_acc, best_macro_f1, best_val_loss,
            os.path.join(config.CHECKPOINT_DIR, 'latest.pth')
        )
        
        print(
            f"Current - Best Acc: {best_acc:.2f}%, "
            f"Best Macro-F1: {best_macro_f1:.4f}, "
            f"Best Val Loss: {best_val_loss:.4f}"
        )
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best accuracy: {best_acc:.2f}% (saved to best_acc.pth)")
    print(f"Best macro-F1: {best_macro_f1:.4f} (saved to best_macro_f1.pth)")
    print(f"Best val loss: {best_val_loss:.4f} (saved to best_loss.pth)")
    print(f"Model saved to: {config.CHECKPOINT_DIR}")


if __name__ == '__main__':
    main()
