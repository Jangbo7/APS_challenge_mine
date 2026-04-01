"""
权重管理工具：只下载 .safetensors 文件到指定目录
"""

import os
import requests
from pathlib import Path
from urllib.parse import urljoin
import json


class WeightManager:
    """管理权重下载和加载"""
    
    # Hugging Face hub 地址（国内可改为 hf-mirror.com）
    HF_MIRROR = "https://hf-mirror.com"
    
    # 模型映射：timm 模型名 -> huggingface 模型 repo
    MODEL_MAPPING = {
        'convnextv2_tiny.fcmae_ft_in22k_in1k': 'timm/convnextv2_tiny.fcmae_ft_in22k_in1k',
        'convnextv2_base.fcmae_ft_in22k_in1k': 'timm/convnextv2_base.fcmae_ft_in22k_in1k',
        'convnextv2_large.fcmae_ft_in22k_in1k': 'timm/convnextv2_large.fcmae_ft_in22k_in1k',
        'convnextv2_huge.fcmae_ft_in22k_in1k': 'timm/convnextv2_huge.fcmae_ft_in22k_in1k',
    }
    
    def __init__(self, cache_dir='./pretrained_weights'):
        self.cache_dir = Path(cache_dir).absolute()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_weight(self, model_name, force=False):
        """
        只下载指定模型的 .safetensors 权重文件
        
        Args:
            model_name: timm 模型名称（如 'convnextv2_large.fcmae_ft_in22k_in1k'）
            force: 是否强制重新下载
        
        Returns:
            本地权重文件路径
        """
        if model_name not in self.MODEL_MAPPING:
            raise ValueError(f"不支持的模型: {model_name}\n支持: {list(self.MODEL_MAPPING.keys())}")
        
        hf_repo = self.MODEL_MAPPING[model_name]
        local_path = self.cache_dir / f"{model_name}.safetensors"
        
        # 如果文件已存在且不强制重新下载，直接返回
        if local_path.exists() and not force:
            print(f"✓ 权重已存在: {local_path}")
            return str(local_path)
        
        # 构建下载 URL
        url = f"{self.HF_MIRROR}/{hf_repo}/resolve/main/model.safetensors"
        
        print(f"📥 下载 {model_name}...")
        print(f"   URL: {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            
            # 下载并保存
            downloaded = 0
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            progress = (downloaded / total_size) * 100
                            print(f"   進度: {progress:.1f}%", end='\r')
            
            print(f"✓ 下载完成: {local_path} ({total_size / (1024**2):.1f} MB)")
            return str(local_path)
        
        except Exception as e:
            if local_path.exists():
                local_path.unlink()  # 删除不完整的文件
            raise RuntimeError(f"下载失败: {e}")
    
    def download_all(self, force=False):
        """下载所有支持的模型权重"""
        print(f"缓存目录: {self.cache_dir}\n")
        
        for model_name in self.MODEL_MAPPING.keys():
            try:
                self.download_weight(model_name, force=force)
            except Exception as e:
                print(f"✗ {model_name} 失败: {e}\n")
    
    def get_local_weight_path(self, model_name):
        """获取本地权重文件路径（如果存在）"""
        local_path = self.cache_dir / f"{model_name}.safetensors"
        if local_path.exists():
            return str(local_path)
        return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='下载预训练权重')
    parser.add_argument('--cache-dir', default='./pretrained_weights', help='缓存目录')
    parser.add_argument('--model', help='指定模型下载（不指定则下载全部）')
    parser.add_argument('--force', action='store_true', help='强制重新下载')
    
    args = parser.parse_args()
    
    manager = WeightManager(cache_dir=args.cache_dir)
    
    if args.model:
        manager.download_weight(args.model, force=args.force)
    else:
        manager.download_all(force=args.force)
