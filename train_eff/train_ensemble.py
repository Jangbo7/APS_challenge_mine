"""
训练多个 EfficientNet 模型用于集成学习

该脚本支持：
1. 训练多个独立的模型
2. 使用不同的随机种子或数据增强策略
3. 保存所有模型用于后续集成
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import get_dataloaders
from model import build_model
from utils import (
    set_seed, save_checkpoint, load_checkpoint,
    AverageMeter, accuracy, log_training, get_current_time
)
from metrics import compute_class_metrics, print_class_metrics, print_confusion_matrix


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
def validate(model, val_loader, criterion, device, num_classes, class_names=None):
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
    
    # 计算每个类别的指标
    metrics_dict = compute_class_metrics(all_preds, all_labels, num_classes, class_names)
    
    return losses.avg, top1.avg, all_preds, all_labels, metrics_dict


def train_single_model(config, model_id, seed=None):
    """
    训练单个模型
    
    Args:
        config: 配置对象
        model_id: 模型ID（用于区分不同模型）
        seed: 随机种子（如果为None，则使用config.SEED + model_id）
    
    Returns:
        best_acc: 最佳准确率
        model_path: 最佳模型路径
    """
    # 设置随机种子
    if seed is None:
        seed = config.SEED + model_id
    set_seed(seed)
    
    print(f"\n{'='*80}")
    print(f"Training Model {model_id} (seed={seed})")
    print(f"{'='*80}\n")
    
    # 设置设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型特定的保存目录
    model_dir = os.path.join(config.CHECKPOINT_DIR, f'model_{model_id}')
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建日志文件
    log_file = os.path.join(model_dir, f"train_{get_current_time()}.log")
    
    # 加载数据
    print("Loading data...")
    train_loader, val_loader, class_names = get_dataloaders(config)
    print(f"Number of classes: {len(class_names)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # 构建模型
    print("Building model...")
    model = build_model(config)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
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
    checkpoint_path = os.path.join(model_dir, 'latest.pth')
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
            val_loss, val_acc, all_preds, all_labels, metrics_dict = validate(
                model, val_loader, criterion, device,
                num_classes=config.NUM_CLASSES,
                class_names=class_names
            )
            
            # 打印类别级指标
            print_class_metrics(metrics_dict, class_names)
        
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
                os.path.join(model_dir, 'best.pth')
            )
        
        # 保存最新检查点
        save_checkpoint(
            model, optimizer, epoch, best_acc,
            os.path.join(model_dir, 'latest.pth')
        )
        
        print(f"Best accuracy so far: {best_acc:.2f}%")
    
    print(f"\n{'='*50}")
    print(f"Model {model_id} training completed! Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {model_dir}")
    print(f"{'='*50}\n")
    
    return best_acc, os.path.join(model_dir, 'best.pth')


def train_multiple_models(base_config=None):
    """
    训练多个模型
    
    Args:
        base_config: 基础配置对象（如果为None，则创建新配置）
    
    Returns:
        results: 包含每个模型训练结果的列表
    """
    if base_config is None:
        base_config = Config()
    
    num_models = base_config.NUM_ENSEMBLE_MODELS
    
    print("\n" + "="*80)
    print(f"Training {num_models} models for ensemble learning")
    print("="*80 + "\n")
    
    results = []
    
    for model_id in range(num_models):
        print(f"\n{'#'*80}")
        print(f"# Model {model_id + 1}/{num_models}")
        print(f"{'#'*80}")
        
        best_acc, model_path = train_single_model(base_config, model_id)
        
        results.append({
            'model_id': model_id,
            'seed': base_config.SEED + model_id,
            'best_acc': best_acc,
            'model_path': model_path
        })
    
    # 打印所有模型的训练结果
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    print(f"{'Model ID':<10} {'Seed':<10} {'Best Acc':<15} {'Model Path'}")
    print("-"*80)
    
    for result in results:
        print(f"{result['model_id']:<10} {result['seed']:<10} {result['best_acc']:.2f}%{'':<10} {result['model_path']}")
    
    avg_acc = np.mean([r['best_acc'] for r in results])
    std_acc = np.std([r['best_acc'] for r in results])
    
    print("-"*80)
    print(f"Average Accuracy: {avg_acc:.2f}% ± {std_acc:.2f}%")
    print("="*80 + "\n")
    
    # 保存训练结果
    results_file = os.path.join(base_config.CHECKPOINT_DIR, 'ensemble_results.txt')
    with open(results_file, 'w') as f:
        f.write("Model Training Results for Ensemble\n")
        f.write("="*80 + "\n")
        f.write(f"{'Model ID':<10} {'Seed':<10} {'Best Acc':<15} {'Model Path'}\n")
        f.write("-"*80 + "\n")
        for result in results:
            f.write(f"{result['model_id']:<10} {result['seed']:<10} {result['best_acc']:.2f}%{'':<10} {result['model_path']}\n")
        f.write("-"*80 + "\n")
        f.write(f"Average Accuracy: {avg_acc:.2f}% ± {std_acc:.2f}%\n")
        f.write("="*80 + "\n")
    
    print(f"Results saved to: {results_file}")
    
    return results


def main():
    """主函数"""
    # 加载配置
    config = Config()
    
    # 训练多个模型
    results = train_multiple_models(base_config=config)
    
    print("\n" + "="*80)
    print("All models training completed!")
    print("="*80 + "\n")
    
    # 如果配置了自动预测，则运行集成预测
    if config.AUTO_PREDICT_ENSEMBLE:
        print("\n" + "="*80)
        print("Auto-running ensemble prediction...")
        print("="*80 + "\n")
        
        # 导入预测模块
        import predict_ensemble
        
        # 调用预测函数
        predict_ensemble.main()


if __name__ == '__main__':
    main()
