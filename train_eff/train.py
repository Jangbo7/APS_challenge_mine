import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from loss import FocalLoss
from config import Config
from dataset import get_dataloaders
from model import build_model
from utils import (
    set_seed, save_checkpoint, load_checkpoint,
    AverageMeter, accuracy, log_training, get_current_time
)
from metrics import (
    compute_class_metrics, print_class_metrics,
    print_confusion_matrix
)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
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
    
    pbar = tqdm(val_loader, desc="[Validate]")
    for images, labels, _ in pbar:
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
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{top1.avg:.2f}%'
        })
    
    return losses.avg, top1.avg, all_preds, all_labels


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
    start_epoch = 0
    
    # 检查是否有检查点可以恢复
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'latest.pth')
    if config.RESUME and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, best_acc = load_checkpoint(model, optimizer, checkpoint_path, device)
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
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # 验证
        val_loss, val_acc = None, None
        if val_loader:
            val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device)
            
            # 计算并打印类别级指标
            metrics_dict = compute_class_metrics(
                all_preds, all_labels, 
                num_classes=config.NUM_CLASSES, 
                class_names=class_names
            )
            print_class_metrics(metrics_dict, class_names)
            print_confusion_matrix(metrics_dict['confusion_matrix'], class_names)
        
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
        is_best = val_acc is not None and val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                os.path.join(config.CHECKPOINT_DIR, 'best.pth')
            )
        
        # 保存最新检查点
        save_checkpoint(
            model, optimizer, epoch, best_acc,
            os.path.join(config.CHECKPOINT_DIR, 'latest.pth')
        )
        
        print(f"Best accuracy so far: {best_acc:.2f}%")
    
    print(f"\n{'='*50}")
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {config.CHECKPOINT_DIR}")


if __name__ == '__main__':
    main()
