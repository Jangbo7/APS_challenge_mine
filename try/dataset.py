import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np


class APSDataset(Dataset):
    def __init__(self, root_dir, class_to_idx=None, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        
        if is_train:
            if class_to_idx is None:
                self.classes = sorted(os.listdir(root_dir))
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            else:
                self.class_to_idx = class_to_idx
                self.classes = list(class_to_idx.keys())
            
            for class_name in self.classes:
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        else:
            self.class_to_idx = class_to_idx
            for img_name in os.listdir(root_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root_dir, img_name)
                    self.samples.append((img_path, -1))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


class APSSubsetDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


def get_transforms(is_train=True, image_size=224):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def split_dataset_stratified(dataset, val_ratio=0.2, seed=42):
    """
    按类别分层分割数据集
    
    Args:
        dataset: 原始数据集
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        train_indices, val_indices: 训练集和验证集的索引
    """
    random.seed(seed)
    
    # 按类别分组索引
    class_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    # 对每个类别进行分割
    for label, indices in class_to_indices.items():
        random.shuffle(indices)
        val_size = int(len(indices) * val_ratio)
        val_indices.extend(indices[:val_size])
        train_indices.extend(indices[val_size:])
    
    return train_indices, val_indices


def get_dataloaders(config):
    """
    获取数据加载器
    
    从训练集中分割验证集，val_noclass 作为无标签测试集
    """
    with open(config.CLASSNAME_FILE, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    # 创建完整训练数据集（不应用transform）
    full_train_dataset = APSDataset(
        root_dir=config.TRAIN_DIR,
        class_to_idx=class_to_idx,
        transform=None,
        is_train=True
    )
    
    print(f"Total training samples: {len(full_train_dataset)}")
    
    # 分层分割训练集和验证集
    train_indices, val_indices = split_dataset_stratified(
        full_train_dataset,
        val_ratio=config.VAL_SPLIT,
        seed=config.SEED
    )
    
    print(f"Training samples after split: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # 创建带不同transform的子数据集
    train_transform = get_transforms(is_train=True, image_size=config.IMAGE_SIZE)
    val_transform = get_transforms(is_train=False, image_size=config.IMAGE_SIZE)
    
    train_dataset = APSSubsetDataset(full_train_dataset, train_indices, train_transform)
    val_dataset = APSSubsetDataset(full_train_dataset, val_indices, val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, class_names


def get_test_dataloader(config):
    """
    获取无标签测试集数据加载器
    """
    with open(config.CLASSNAME_FILE, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    test_transform = get_transforms(is_train=False, image_size=config.IMAGE_SIZE)
    test_dataset = APSDataset(
        root_dir=config.TEST_DIR,
        class_to_idx=class_to_idx,
        transform=test_transform,
        is_train=False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader, class_names
