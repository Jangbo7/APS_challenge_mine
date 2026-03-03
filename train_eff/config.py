import os

class Config:
    # 数据路径
    # DATA_ROOT = r"c:\Users\jangb\Desktop\contest_group\data\APS_dataset"
    DATA_DIR = "/data2/jiangwb/eff/data/APS_dataset"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    TEST_DIR = os.path.join(DATA_ROOT, "val_noclass")  # 无标签测试集
    CLASSNAME_FILE = os.path.join(DATA_ROOT, "classname.txt")
    
    # 模型保存路径
    # CHECKPOINT_DIR = r"c:\Users\jangb\Desktop\contest_group\train_eff\checkpoints2"
    CHECKPOINT_DIR = "/data2/jiangwb/eff/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 训练参数
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # 图像参数
    IMAGE_SIZE = 224
    NUM_CLASSES = 17
    
    # 设备
    DEVICE = "cuda"
    
    # 随机种子
    SEED = 42
    
    # 验证集比例（从训练集分割）
    VAL_SPLIT = 0.2
    
    # 是否对训练集进行下采样（每个类别最多使用100个样本）
    IF_UNDERAMPLE = True
    
    # 是否使用均衡BatchSampler（每个batch中各类别样本比例为1:1）
    # 当为True时，batch_size会自动调整为类别数的倍数
    IF_OVERSAMPLE = True
