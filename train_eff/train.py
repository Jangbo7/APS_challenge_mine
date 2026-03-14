import os
import sys

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
from loss import FocalLoss
from config import Config
from dataset import get_dataloaders
from model import build_model
from augmentation import mixup_data, cutmix_data, mixed_criterion
from utils import (
    set_seed, save_checkpoint, load_checkpoint,
    AverageMeter, accuracy, log_training, get_current_time,
    save_error_samples
)
from metrics import (
    compute_class_metrics, print_class_metrics,
    print_confusion_matrix
)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,config):
    """训练一个 epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    

    aug_type = getattr(config, 'AUG_TYPE', 'none')
    aug_alpha = getattr(config, 'AUG_ALPHA', 1.0)
    aug_prob = getattr(config, 'AUG_PROB', 0.5)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        # print(device)
                # ---- batch 级数据增强 ----
        use_aug = (aug_type != 'none') and (np.random.rand() < aug_prob)
        if use_aug:
            if aug_type == 'both':
                # 随机选一种
                chosen = np.random.choice(['mixup', 'cutmix'])
            else:
                chosen = aug_type

            if chosen == 'mixup':
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=aug_alpha)
            else:  # cutmix
                images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=aug_alpha)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixed_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
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
