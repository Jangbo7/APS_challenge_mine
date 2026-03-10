import os

class Config:
    # 数据路径
    # DATA_ROOT = r"c:\Users\jangb\Desktop\contest_group\data\APS_dataset"
    DATA_ROOT = "/data2/jiangwb/eff/data/APS_dataset"
    # DATA_ROOT = "data/APS_dataset"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    TEST_DIR = os.path.join(DATA_ROOT, "val_noclass")  # 无标签测试集
    CLASSNAME_FILE = os.path.join(DATA_ROOT, "classname.txt")
    
    # 模型保存路径
    # CHECKPOINT_DIR = r"c:\Users\jangb\Desktop\contest_group\train_eff\checkpoints2"
    CHECKPOINT_DIR = "/data2/jiangwb/eff/checkpoints"
    # CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 预训练权重下载缓存路径
    # 默认为 ~/.cache/torch/hub/checkpoints
    # 可以修改为自定义路径，例如: "/data2/jiangwb/eff/pretrained_weights"
    PRETRAINED_WEIGHTS_DIR = "/data2/jiangwb/eff/pretrained_weights"  # 设为 None 时使用 PyTorch 默认路径
    
    # 训练参数
    BATCH_SIZE = 32
    NUM_EPOCHS = 35
    LEARNING_RATE = 0.8e-4
    WEIGHT_DECAY = 0.8e-4
    
    # 图像参数
    IMAGE_SIZE = 224
    NUM_CLASSES = 17
    
    # 频率通道处理
    USE_FREQ_CHANNELS = False  # 是否使用频率通道(高频+低频)特征，如果为True输出5通道，为False输出3通道
    LOW_PASS_SIZE = 12         # 低频掩码大小(仅当USE_FREQ_CHANNELS=True时有效)
    
    # 设备
    DEVICE = "cuda"
    
    # 随机种子
    SEED = 42
    
    # 验证集比例（从训练集分割）
    VAL_SPLIT = 0.2
    
    # 模型类型选择
    # EfficientNetV2: 'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l'
    # ConvNeXt V1: 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'
    # ConvNeXtV2: 'convnextv2_tiny', 'convnextv2_base', 'convnextv2_large', 'convnextv2_huge' (需要安装 timm: pip install timm)
    MODEL_TYPE = 'convnextv2_base'
    
    # 是否对训练集进行下采样（每个类别最多使用200个样本）
    IF_UNDERAMPLE = False
    undersample_num = 250
    
    # 是否使用均衡BatchSampler（每个batch中各类别样本比例为1:1）
    # 当为True时，batch_size会自动调整为类别数的倍数
    IF_OVERSAMPLE = True
    
    # 集成学习配置
    # 要训练的模型数量
    NUM_ENSEMBLE_MODELS = 10
    # 是否在训练完成后自动运行集成预测
    AUTO_PREDICT_ENSEMBLE = True
    # 集成预测策略: 'average' (平均), 'voting' (投票), 'weighted' (加权)
    ENSEMBLE_STRATEGY = 'voting'
    
    # 是否自动恢复训练（从最新的检查点继续）
    RESUME = False

    # 损失函数配置
    LOSS_TYPE = 'focal'          # 'cross_entropy' | 'focal'
    FOCAL_GAMMA = 2.0            # focal loss 聚焦参数
    LABEL_SMOOTHING = 0.1        # 标签平滑
    USE_CLASS_ALPHA = False      # 是否按类别频率自动计算 alpha 权重