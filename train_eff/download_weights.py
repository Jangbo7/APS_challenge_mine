#!/usr/bin/env python3
"""
下载预训练权重到本地缓存
在登录节点执行的脚本
"""

import os
import sys
import timm
import argparse
from pathlib import Path

def download_weights(cache_dir=None, models=None):
    """下载指定的预训练权重"""
    
    # 设置缓存目录
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = os.path.abspath(cache_dir)
        os.environ['TORCH_HOME'] = os.path.abspath(cache_dir)
        print(f"✓ 缓存目录: {os.path.abspath(cache_dir)}")
    
    # 默认模型列表
    if models is None:
        models = [
            'convnextv2_tiny.fcmae_ft_in22k_in1k',
            'convnextv2_base.fcmae_ft_in22k_in1k',
            'convnextv2_large.fcmae_ft_in22k_in1k',
            'convnextv2_huge.fcmae_ft_in22k_in1k'
        ]
    
    print(f"\n开始下载 {len(models)} 个模型...\n")
    
    success_count = 0
    failed_models = []
    
    for i, model_name in enumerate(models, 1):
        try:
            print(f"[{i}/{len(models)}] 下载 {model_name}...", end=" ", flush=True)
            model = timm.create_model(model_name, pretrained=True)
            print("✓ 成功")
            success_count += 1
        except Exception as e:
            print(f"✗ 失败")
            print(f"       错误: {str(e)[:100]}")
            failed_models.append((model_name, str(e)))
    
    print(f"\n{'='*60}")
    print(f"下载完成: {success_count}/{len(models)} 成功")
    
    if failed_models:
        print(f"\n失败的模型:")
        for model_name, error in failed_models:
            print(f"  - {model_name}")
            print(f"    {error[:80]}...")
    
    print(f"{'='*60}\n")
    
    # 显示缓存目录大小
    if cache_dir and os.path.exists(cache_dir):
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(cache_dir)
            for filename in filenames
        )
        print(f"缓存总大小: {total_size / (1024**3):.2f} GB")
    
    return len(failed_models) == 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='下载预训练权重')
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./pretrained_weights',
        help='权重缓存目录 (默认: ./pretrained_weights)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='指定要下载的模型 (默认: 全部 convnextv2 模型)'
    )
    
    args = parser.parse_args()
    
    success = download_weights(cache_dir=args.cache_dir, models=args.models)
    sys.exit(0 if success else 1)
