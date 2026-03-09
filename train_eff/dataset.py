import os
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, Sampler, BatchSampler
from torchvision import transforms
from PIL import Image
import numpy as np


class BalancedBatchSampler(BatchSampler):
    """
    均衡BatchSampler - 确保每个batch中各类别样本比例为1:1
    
    工作原理：
    1. 按类别分组索引
    2. 计算每个batch中每个类别应包含的样本数 n
    3. 正常情况下，每个batch包含所有类别，每个类别 n 个样本
    4. 样本较少的类别允许过采样（循环使用）
    
    注意：返回的索引是相对于子集的索引（0到len(indices)-1），而不是原始数据集的索引
    """
    def __init__(self, dataset, indices, batch_size, num_classes, seed=42, use_oversampling=True):
        """
        Args:
            dataset: 数据集对象
            indices: 训练集索引列表（原始数据集的索引）
            batch_size: 期望的batch大小（会被调整为类别数的倍数）
            num_classes: 类别数量
            seed: 随机种子
            use_oversampling: 是否允许过采样（默认True）
                              为True时，按照最大类别样本数决定batch数量，其他类别过采样
                              为False时，按照最小类别样本数决定batch数量
        """
        self.dataset = dataset
        self.indices = indices
        self.num_classes = num_classes
        self.seed = seed
        self.use_oversampling = use_oversampling
        
        # 计算每个batch中每个类别应包含的样本数 n
        # batch_size = n * num_classes
        self.n_per_class = max(1, batch_size // num_classes)
        self.actual_batch_size = self.n_per_class * num_classes
        
        # 按类别分组索引（存储的是子集中的位置索引，而不是原始索引）
        self.class_to_subset_indices = {i: [] for i in range(num_classes)}
        for subset_idx, original_idx in enumerate(indices):
            _, label = dataset.samples[original_idx]
            if label in self.class_to_subset_indices:
                self.class_to_subset_indices[label].append(subset_idx)
        
        # 计算每个类别的样本数
        self.class_sizes = {label: len(indices) for label, indices in self.class_to_subset_indices.items()}
        
        # 计算最小和最大类别的样本数
        self.min_class_size = min(self.class_sizes.values())
        self.max_class_size = max(self.class_sizes.values())
        
        # 根据是否使用过采样决定使用哪个类别大小来计算batch数量
        if use_oversampling:
            # 使用过采样：按照最大类别样本数决定batch数量，其他类别循环使用
            reference_size = self.max_class_size
        else:
            # 不使用过采样：按照最小类别样本数决定batch数量
            reference_size = self.min_class_size
        
        # 计算每个epoch的batch数量
        self.batches_per_epoch = reference_size // self.n_per_class
        
        # 如果batch数量太少（少于10个），调整 n_per_class 以获得更多batch
        min_batches_per_epoch = 10
        if self.batches_per_epoch < min_batches_per_epoch:
            self.n_per_class = max(1, reference_size // min_batches_per_epoch)
            self.actual_batch_size = self.n_per_class * num_classes
            self.batches_per_epoch = reference_size // self.n_per_class
            
            if self.batches_per_epoch == 0:
                self.batches_per_epoch = 1
        
        # 确保至少有1个batch
        if self.batches_per_epoch == 0:
            self.n_per_class = reference_size
            self.actual_batch_size = self.n_per_class * num_classes
            self.batches_per_epoch = 1
        
        self.epoch = 0
    
    def __iter__(self):
        """生成均衡的batch"""
        random.seed(self.seed + self.epoch)
        
        # 为每个类别创建样本列表的副本并打乱
        class_indices_shuffled = {}
        for label, indices in self.class_to_subset_indices.items():
            indices_copy = indices.copy()
            random.shuffle(indices_copy)
            class_indices_shuffled[label] = indices_copy
        
        # 记录每个类别当前使用的样本索引
        class_current_idx = {label: 0 for label in range(self.num_classes)}
        
        # 生成batch
        for batch_idx in range(self.batches_per_epoch):
            batch = []
            
            # 检查是否是最后一个batch
            is_last_batch = (batch_idx == self.batches_per_epoch - 1)
            
            # 检查剩余样本是否足够组成一个完整的batch
            remaining_samples = sum(
                len(class_indices_shuffled[label]) - class_current_idx[label]
                for label in range(self.num_classes)
            )
            
            if is_last_batch and remaining_samples < self.actual_batch_size:
                # 最后一个batch样本不足，随机选择部分类别来填充
                # 计算需要多少个类别
                needed_classes = remaining_samples // self.n_per_class
                if remaining_samples % self.n_per_class != 0:
                    needed_classes += 1
                needed_classes = max(1, min(needed_classes, self.num_classes))
                
                # 随机选择 needed_classes 个类别
                available_classes = [label for label in range(self.num_classes)
                                    if len(class_indices_shuffled[label]) > 0]
                # 确保 needed_classes 不超过 available_classes 的数量
                needed_classes = min(needed_classes, len(available_classes))
                selected_classes = random.sample(available_classes, needed_classes)
            else:
                # 正常情况：使用所有类别
                selected_classes = list(range(self.num_classes))
            
            for label in selected_classes:
                # 从该类别中取出 n_per_class 个样本
                indices_for_batch = []
                for _ in range(self.n_per_class):
                    # 检查是否还有剩余样本
                    if class_current_idx[label] < len(class_indices_shuffled[label]):
                        indices_for_batch.append(class_indices_shuffled[label][class_current_idx[label]])
                        class_current_idx[label] += 1
                    else:
                        # 样本不足，循环使用
                        idx = class_current_idx[label] % len(class_indices_shuffled[label])
                        indices_for_batch.append(class_indices_shuffled[label][idx])
                        class_current_idx[label] += 1
                
                batch.extend(indices_for_batch)
            
            # 打乱batch内的顺序
            random.shuffle(batch)
            yield batch
        
        self.epoch += 1
    
    def __len__(self):
        """返回每个epoch的batch数量"""
        return self.batches_per_epoch
    
    def get_info(self):
        """获取采样器信息"""
        return {
            'num_classes': self.num_classes,
            'n_per_class': self.n_per_class,
            'actual_batch_size': self.actual_batch_size,
            'batches_per_epoch': self.batches_per_epoch,
            'min_class_size': self.min_class_size,
            'max_class_size': max(self.class_sizes.values())
        }


def calculate_optimal_batch_size(target_batch_size, num_classes):
    """
    计算最优的batch size，使其最接近目标值且是类别数的倍数
    
    Args:
        target_batch_size: 目标batch size
        num_classes: 类别数量
    
    Returns:
        optimal_batch_size: 最优batch size
        n_per_class: 每个类别的样本数
    """
    # 计算最接近的n值
    n = round(target_batch_size / num_classes)
    n = max(1, n)  # 至少每个类别1个样本
    
    optimal_batch_size = n * num_classes
    
    return optimal_batch_size, n


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


class AddFreqChannels(nn.Module):
    """
    输入:  x [3,H,W] (ToTensor后, 0~1)
    输出:  [5,H,W] = RGB(标准ImageNet归一化) + High(1ch) + Low(1ch)
    说明:
    - 高频/低频用 FFT 的频域掩码分离
    - 额外通道做"每样本标准化"(均值0方差1)，避免尺度不稳
    """
    def __init__(self, low_pass_size: int = 12, use_abs: bool = True, eps: float = 1e-6):
        super().__init__()
        self.low_pass_size = int(low_pass_size)
        self.use_abs = bool(use_abs)
        self.eps = float(eps)

        # ImageNet RGB normalize
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _standardize(self, t: torch.Tensor) -> torch.Tensor:
        # t: [1,H,W]
        m = t.mean()
        s = t.std(unbiased=False)
        return (t - m) / (s + self.eps)

    def _fft_split(self, x: torch.Tensor) -> tuple:
        # x: [3,H,W]
        _, H, W = x.shape

        # FFT -> shift to center
        X = torch.fft.fft2(x, dim=(-2, -1))
        X = torch.fft.fftshift(X, dim=(-2, -1))

        mask = torch.zeros((H, W), dtype=torch.float32, device=x.device)
        crow, ccol = H // 2, W // 2
        s = self.low_pass_size
        mask[max(0, crow - s):min(H, crow + s), max(0, ccol - s):min(W, ccol + s)] = 1.0
        mask = mask.view(1, H, W)

        low_fft = X * mask
        high_fft = X * (1.0 - mask)

        # inverse shift -> iFFT
        low = torch.fft.ifftshift(low_fft, dim=(-2, -1))
        low = torch.fft.ifft2(low, dim=(-2, -1)).real

        high = torch.fft.ifftshift(high_fft, dim=(-2, -1))
        high = torch.fft.ifft2(high, dim=(-2, -1)).real

        low1 = low.mean(dim=0, keepdim=True)
        high1 = high.mean(dim=0, keepdim=True)

        if self.use_abs:
            low1 = low1.abs()
            high1 = high1.abs()

        return high1, low1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [3,H,W], float
        if x.ndim != 3 or x.shape[0] != 3:
            raise ValueError(f"AddFreqChannels expects [3,H,W], got {tuple(x.shape)}")

        # RGB normalize
        rgb = (x - self.rgb_mean.to(x.device)) / self.rgb_std.to(x.device)

        # freq channels
        high1, low1 = self._fft_split(x)
        high1 = self._standardize(high1)
        low1 = self._standardize(low1)

        return torch.cat([rgb, high1, low1], dim=0)  # [5,H,W]


def get_transforms(is_train=True, image_size=224, use_freq_channels=False, low_pass_size=12):
    """
    获取图像变换
    
    Args:
        is_train: 是否为训练模式
        image_size: 图像尺寸
        use_freq_channels: 是否使用频率通道处理(添加高频和低频)
                          如果为True，输出5通道 [RGB + High + Low]
                          如果为False，输出3通道 [RGB]
        low_pass_size: 低频掩码大小（仅当use_freq_channels=True时有效）
    """
    if is_train:
        transforms_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ]
        
        if use_freq_channels:
            # 使用频率处理，跳过标准归一化
            transforms_list.append(AddFreqChannels(low_pass_size=low_pass_size))
        else:
            # 使用标准归一化
            transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        return transforms.Compose(transforms_list)
    else:
        transforms_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        
        if use_freq_channels:
            # 使用频率处理，跳过标准归一化
            transforms_list.append(AddFreqChannels(low_pass_size=low_pass_size))
        else:
            # 使用标准归一化
            transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        return transforms.Compose(transforms_list)


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


def undersample_train_set(full_train_dataset, train_indices, max_samples_per_class=200, seed=42):
    """
    对训练集进行下采样，每个类别最多使用指定数量的样本
    
    Args:
        full_train_dataset: 完整训练数据集
        train_indices: 训练集索引
        max_samples_per_class: 每个类别最多使用的样本数
        seed: 随机种子
    
    Returns:
        undersampled_train_indices: 下采样后的训练集索引
        class_stats: 每个类别的统计信息
    """
    random.seed(seed)
    
    # 按类别分组训练集索引
    class_to_train_indices = {}
    for idx in train_indices:
        _, label = full_train_dataset.samples[idx]
        if label not in class_to_train_indices:
            class_to_train_indices[label] = []
        class_to_train_indices[label].append(idx)
    
    # 对每个类别进行下采样
    undersampled_train_indices = []
    class_stats = {}
    
    for label, indices in class_to_train_indices.items():
        original_count = len(indices)
        
        if original_count > max_samples_per_class:
            # 随机采样 max_samples_per_class 个样本
            sampled_indices = random.sample(indices, max_samples_per_class)
            sampled_count = max_samples_per_class
        else:
            # 样本数不足，使用全部样本
            sampled_indices = indices
            sampled_count = original_count
        
        undersampled_train_indices.extend(sampled_indices)
        
        class_stats[label] = {
            'original': original_count,
            'sampled': sampled_count,
            'ratio': sampled_count / original_count if original_count > 0 else 0
        }
    
    return undersampled_train_indices, class_stats


def get_dataloaders(config):
    """
    获取数据加载器
    
    从训练集中分割验证集，val_noclass 作为无标签测试集
    支持对训练集进行下采样（每个类别最多100个样本）
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
    
    # 如果启用下采样，对训练集进行下采样
    if config.IF_UNDERAMPLE:
        print("\n" + "="*60)
        print("Undersampling training set (max 100 samples per class)")
        print("="*60)
        
        train_indices, class_stats = undersample_train_set(
            full_train_dataset,
            train_indices,
            max_samples_per_class=config.undersample_num,
            seed=config.SEED
        )
        
        print(f"Training samples after undersampling: {len(train_indices)}")
        print("\nClass-wise sampling statistics:")
        for label_idx, stats in sorted(class_stats.items()):
            class_name = class_names[label_idx]
            print(f"  {class_name}: {stats['original']} -> {stats['sampled']} samples ({stats['ratio']*100:.1f}%)")
        print("="*60 + "\n")
    
    # 创建带不同transform的子数据集
    train_transform = get_transforms(
        is_train=True,
        image_size=config.IMAGE_SIZE,
        use_freq_channels=getattr(config, "USE_FREQ_CHANNELS", False),
        low_pass_size=getattr(config, "LOW_PASS_SIZE", 12),
    )
    val_transform = get_transforms(
        is_train=False,
        image_size=config.IMAGE_SIZE,
        use_freq_channels=getattr(config, "USE_FREQ_CHANNELS", False),
        low_pass_size=getattr(config, "LOW_PASS_SIZE", 12),
    )
    
    train_dataset = APSSubsetDataset(full_train_dataset, train_indices, train_transform)
    val_dataset = APSSubsetDataset(full_train_dataset, val_indices, val_transform)
    
    # 创建训练数据加载器
    if config.IF_OVERSAMPLE:
        # 使用均衡BatchSampler
        print("\n" + "="*60)
        print("Using Balanced Batch Sampler (1:1 class ratio in each batch)")
        print("="*60)
        
        # 计算最优batch size
        optimal_batch_size, n_per_class = calculate_optimal_batch_size(
            config.BATCH_SIZE, 
            config.NUM_CLASSES
        )
        
        # 创建均衡BatchSampler
        balanced_sampler = BalancedBatchSampler(
            dataset=full_train_dataset,
            indices=train_indices,
            batch_size=config.BATCH_SIZE,
            num_classes=config.NUM_CLASSES,
            seed=config.SEED
        )
        
        sampler_info = balanced_sampler.get_info()
        print(f"Target batch size: {config.BATCH_SIZE}")
        print(f"Actual batch size: {sampler_info['actual_batch_size']} ({sampler_info['n_per_class']} per class × {config.NUM_CLASSES} classes)")
        print(f"Batches per epoch: {sampler_info['batches_per_epoch']}")
        print(f"Min class size in training set: {sampler_info['min_class_size']}")
        print("="*60 + "\n")
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=balanced_sampler,  # 使用batch_sampler而不是batch_size
            num_workers=4,
            pin_memory=True
        )
    else:
        # 使用标准DataLoader
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
    
    test_transform = get_transforms(
        is_train=False,
        image_size=config.IMAGE_SIZE,
        use_freq_channels=getattr(config, "USE_FREQ_CHANNELS", False),
        low_pass_size=getattr(config, "LOW_PASS_SIZE", 12),
    )
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
