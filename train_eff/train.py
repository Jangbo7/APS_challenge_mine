import os
import math
import torch
from torchvision.utils import save_image
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
from augmentation import mixup_data, cutmix_data, occamix_data, mixed_criterion
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


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
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

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images_orig = images.clone()  # 用于可视化对照
        labels_a, labels_b, lam = labels, labels, 1.0
        aug_applied = False

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
                # 随机选一种（包含 occamix）
                chosen = np.random.choice(['mixup', 'cutmix', 'occamix'])
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
            else:
                raise ValueError(f"Unknown augmentation type: {chosen}")

            aug_applied = True

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)

        # 计算损失
        if chosen == 'occamix' and config.LOSS_TYPE == 'cross_entropy' and isinstance(lam, torch.Tensor):
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
        elif chosen == 'occamix' and isinstance(lam, torch.Tensor):
            # focal 维持现有接口：退化为 batch 平均比例
            lam_scalar = float(lam.mean().item())
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
    
    return losses.avg, top1.avg


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
    start_epoch = 0
    
    # 检查是否有检查点可以恢复
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'latest.pth')
    if config.RESUME and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, best_acc, best_macro_f1 = load_checkpoint(model, optimizer, checkpoint_path, device)
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
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1, config
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
        
        if is_best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, best_acc, best_macro_f1,
                os.path.join(config.CHECKPOINT_DIR, 'best_acc.pth')
            )
            print(f"✓ Best accuracy updated: {best_acc:.2f}%")
        
        if is_best_macro_f1:
            best_macro_f1 = val_macro_f1
            save_checkpoint(
                model, optimizer, epoch, best_acc, best_macro_f1,
                os.path.join(config.CHECKPOINT_DIR, 'best_macro_f1.pth')
            )
            print(f"✓ Best macro-F1 updated: {best_macro_f1:.4f}")
        
        # 保存最新检查点
        save_checkpoint(
            model, optimizer, epoch, best_acc, best_macro_f1,
            os.path.join(config.CHECKPOINT_DIR, 'latest.pth')
        )
        
        print(f"Current - Best Acc: {best_acc:.2f}%, Best Macro-F1: {best_macro_f1:.4f}")
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best accuracy: {best_acc:.2f}% (saved to best_acc.pth)")
    print(f"Best macro-F1: {best_macro_f1:.4f} (saved to best_macro_f1.pth)")
    print(f"Model saved to: {config.CHECKPOINT_DIR}")


if __name__ == '__main__':
    main()
