"""
集成多个模型的预测结果

支持多种集成策略：
1. 平均预测概率
2. 投票
3. 加权平均（根据验证集准确率）
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd

# 添加项目路径
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import get_dataloaders, get_test_dataloader
from model import build_model
from utils import load_checkpoint


def load_models(config, model_paths, device):
    """
    加载多个模型
    
    Args:
        config: 配置对象
        model_paths: 模型路径列表
        device: 设备
    
    Returns:
        models: 加载的模型列表
    """
    models = []
    
    for i, model_path in enumerate(model_paths):
        print(f"Loading model {i+1}/{len(model_paths)}: {model_path}")
        
        model = build_model(config)
        model = model.to(device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models.append(model)
    
    print(f"Successfully loaded {len(models)} models")
    return models


def ensemble_predict_average(models, dataloader, device):
    """
    平均预测概率集成
    
    Args:
        models: 模型列表
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        all_preds: 预测标签列表
        all_probs: 预测概率列表
        img_paths: 图片路径列表
    """
    all_probs = []
    all_preds = []
    img_paths = []
    
    with torch.no_grad():
        for images, _, paths in tqdm(dataloader, desc="Ensemble predicting"):
            images = images.to(device)
            
            # 收集所有模型的预测概率
            batch_probs = []
            for model in models:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())
            
            # 平均所有模型的预测概率
            avg_probs = np.mean(batch_probs, axis=0)
            preds = np.argmax(avg_probs, axis=1)
            
            all_probs.extend(avg_probs)
            all_preds.extend(preds)
            img_paths.extend(paths)
    
    return all_preds, all_probs, img_paths


def ensemble_predict_voting(models, dataloader, device):
    """
    投票集成
    
    Args:
        models: 模型列表
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        all_preds: 预测标签列表
        all_probs: 预测概率列表（投票比例）
        img_paths: 图片路径列表
    """
    all_preds = []
    all_probs = []
    img_paths = []
    
    with torch.no_grad():
        for images, _, paths in tqdm(dataloader, desc="Voting ensemble"):
            images = images.to(device)
            
            # 收集所有模型的预测
            batch_preds = []
            for model in models:
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                batch_preds.append(preds)
            
            # 投票
            batch_preds = np.array(batch_preds)  # shape: (num_models, batch_size)
            final_preds = []
            final_probs = []
            
            for i in range(batch_preds.shape[1]):
                votes = batch_preds[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                
                # 选择票数最多的类别
                final_pred = unique[np.argmax(counts)]
                final_preds.append(final_pred)
                
                # 计算投票比例作为概率
                prob = np.zeros(models[0].num_classes if hasattr(models[0], 'num_classes') else 17)
                for cls, count in zip(unique, counts):
                    prob[cls] = count / len(models)
                final_probs.append(prob)
            
            all_preds.extend(final_preds)
            all_probs.extend(final_probs)
            img_paths.extend(paths)
    
    return all_preds, all_probs, img_paths


def ensemble_predict_weighted(models, dataloader, device, weights=None):
    """
    加权平均集成
    
    Args:
        models: 模型列表
        dataloader: 数据加载器
        device: 设备
        weights: 模型权重列表（如果为None，则使用等权重）
    
    Returns:
        all_preds: 预测标签列表
        all_probs: 预测概率列表
        img_paths: 图片路径列表
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    # 归一化权重
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    all_probs = []
    all_preds = []
    img_paths = []
    
    with torch.no_grad():
        for images, _, paths in tqdm(dataloader, desc="Weighted ensemble"):
            images = images.to(device)
            
            # 收集所有模型的预测概率
            batch_probs = []
            for model in models:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())
            
            # 加权平均
            weighted_probs = np.zeros_like(batch_probs[0])
            for prob, weight in zip(batch_probs, weights):
                weighted_probs += prob * weight
            
            preds = np.argmax(weighted_probs, axis=1)
            
            all_probs.extend(weighted_probs)
            all_preds.extend(preds)
            img_paths.extend(paths)
    
    return all_preds, all_probs, img_paths


def save_predictions(img_paths, preds, class_names, output_file):
    """
    保存预测结果到txt文件
    格式：每行 "图片名 预测标签"
    """
    # 提取文件名
    filenames = [os.path.basename(path) for path in img_paths]
    
    # 保存为txt文件（与test.py格式一致）
    with open(output_file, 'w', encoding='utf-8') as f:
        for filename, pred in zip(filenames, preds):
            line = f"{filename} {pred}\n"
            f.write(line)
    
    print(f"Predictions saved to: {output_file}")


def main():
    """主函数"""
    # 加载配置
    config = Config()
    
    # 设置设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 确定要加载的检查点文件名
    checkpoint_name = f"{config.ENSEMBLE_CHECKPOINT}.pth" if hasattr(config, 'ENSEMBLE_CHECKPOINT') else 'best_acc.pth'
    
    # 查找所有训练好的模型
    model_dirs = []
    for item in os.listdir(config.CHECKPOINT_DIR):
        item_path = os.path.join(config.CHECKPOINT_DIR, item)
        if os.path.isdir(item_path) and item.startswith('model_'):
            model_path = os.path.join(item_path, checkpoint_name)
            if os.path.exists(model_path):
                model_dirs.append(model_path)
    
    if len(model_dirs) == 0:
        print(f"No trained models found! Please run train_ensemble.py first.")
        print(f"(Looking for '{checkpoint_name}' in {config.CHECKPOINT_DIR})")
        return
    
    print(f"\nFound {len(model_dirs)} trained models:")
    for i, path in enumerate(model_dirs):
        print(f"  {i+1}. {path}")
    
    # 加载模型
    print("\nLoading models...")
    models = load_models(config, model_dirs, device)
    
    # 加载测试数据
    print("\nLoading test data...")
    test_loader, class_names = get_test_dataloader(config)
    
    # 使用配置中的集成策略
    strategy = config.ENSEMBLE_STRATEGY.lower()
    
    if strategy == 'voting':
        print("\nUsing Voting ensemble...")
        preds, probs, img_paths = ensemble_predict_voting(models, test_loader, device)
    elif strategy == 'weighted':
        print("\nUsing Weighted ensemble...")
        # 可以根据验证集准确率设置权重
        # 这里使用等权重作为示例
        weights = None
        preds, probs, img_paths = ensemble_predict_weighted(models, test_loader, device, weights)
    else:  # 'average' or default
        print("\nUsing Average ensemble...")
        preds, probs, img_paths = ensemble_predict_average(models, test_loader, device)
    
    # 保存预测结果（txt格式，与test.py一致）
    output_file = os.path.join(config.CHECKPOINT_DIR, 'result.txt')
    save_predictions(img_paths, preds, class_names, output_file)
    
    # 显示前10个预测结果
    print("\nFirst 10 predictions:")
    for i, (path, pred) in enumerate(zip(img_paths[:10], preds[:10])):
        print(f"  {os.path.basename(path)} {pred}")
    
    print("\n" + "="*80)
    print("Ensemble prediction completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
