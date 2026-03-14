import os
import torch
import random
import numpy as np
from datetime import datetime
from shutil import copy2
from pathlib import Path


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_current_time():
    """获取当前时间字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_checkpoint(model, optimizer, epoch, best_acc, best_macro_f1=None, filepath=None):
    """保存模型检查点"""
    # 处理DataParallel模型
    model_state = model.state_dict()
    if isinstance(model, torch.nn.DataParallel):
        # 移除 "module." 前缀
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'best_macro_f1': best_macro_f1 if best_macro_f1 is not None else 0.0,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, device='cuda'):
    """加载模型检查点"""
    checkpoint = torch.load(filepath, map_location=device)
    model_state = checkpoint['model_state_dict']
    
    # 处理DataParallel模型的加载
    if isinstance(model, torch.nn.DataParallel):
        # 如果state_dict没有"module."前缀，添加它
        if not any(k.startswith('module.') for k in model_state.keys()):
            model_state = {f'module.{k}': v for k, v in model_state.items()}
        model.load_state_dict(model_state)
    else:
        # 如果state_dict有"module."前缀，移除它
        if any(k.startswith('module.') for k in model_state.keys()):
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        model.load_state_dict(model_state)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    best_macro_f1 = checkpoint.get('best_macro_f1', 0.0)
    print(f"Checkpoint loaded: {filepath}, Epoch: {epoch}, Best Acc: {best_acc:.4f}, Best Macro-F1: {best_macro_f1:.4f}")
    return epoch, best_acc, best_macro_f1


class AverageMeter:
    """计算和存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """计算 top-k 准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def log_training(epoch, num_epochs, train_loss, train_acc, val_loss=None, val_acc=None, log_file=None):
    """记录训练日志"""
    log_str = f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
    if val_loss is not None and val_acc is not None:
        log_str += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
    
    print(log_str)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')


def save_error_samples(epoch, error_sample_list, error_samples_dir, class_names=None):
    """
    保存验证错误的样本图片
    
    Args:
        epoch: 当前epoch编号
        error_sample_list: 错误样本列表，每个元素是字典：
            {
                'img_path': str,           # 原始图片路径
                'pred_label': int,         # 预测标签
                'true_label': int,         # 真实标签
                'confidence': float        # 预测的最大概率
            }
        error_samples_dir: 错误样本保存的母目录
        class_names: 类别名称列表（用于生成更可读的文件名）
    """
    if not error_sample_list:
        return
    
    # 创建epoch目录
    epoch_dir = os.path.join(error_samples_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # 创建汇总CSV文件
    csv_file = os.path.join(epoch_dir, 'error_summary.csv')
    
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write('Image_Name,True_Label,Predicted_Label,Confidence\n')
        
        for idx, sample in enumerate(error_sample_list, 1):
            img_path = sample['img_path']
            pred_label = sample['pred_label']
            true_label = sample['true_label']
            confidence = sample['confidence']
            
            # 获取原始文件名
            img_filename = os.path.basename(img_path)
            filename_without_ext = os.path.splitext(img_filename)[0]
            file_ext = os.path.splitext(img_filename)[1]
            
            # 生成新文件名
            if class_names:
                true_class = class_names[true_label] if true_label < len(class_names) else f"class_{true_label}"
                pred_class = class_names[pred_label] if pred_label < len(class_names) else f"class_{pred_label}"
                new_filename = f"{idx:04d}_{filename_without_ext}_true[{true_class}]_pred[{pred_class}]_{confidence:.4f}{file_ext}"
            else:
                new_filename = f"{idx:04d}_{filename_without_ext}_true[{true_label}]_pred[{pred_label}]_{confidence:.4f}{file_ext}"
            
            # 复制图片文件
            dest_path = os.path.join(epoch_dir, new_filename)
            try:
                copy2(img_path, dest_path)
            except Exception as e:
                print(f"Warning: Failed to copy {img_path} to {dest_path}: {e}")
            
            # 写入CSV记录
            if class_names:
                f.write(f"{new_filename},{class_names[true_label]},{class_names[pred_label]},{confidence:.4f}\n")
            else:
                f.write(f"{new_filename},{true_label},{pred_label},{confidence:.4f}\n")
    
    print(f"✓ Saved {len(error_sample_list)} error samples to: {epoch_dir}")
