import os

class Config:
    # 数据路径
    # DATA_ROOT = r"c:\Users\jangb\Desktop\contest_group\data\APS_dataset"
    # DATA_ROOT = "/data2/jiangwb/eff/data/APS_dataset"
    DATA_ROOT = "data/APS_dataset"
    # DATA_ROOT = "../data/APS_dataset"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    # TEST_DIR = os.path.join(DATA_ROOT, "val_noclass")  # 无标签测试集
    TEST_DIR = os.path.join(DATA_ROOT, "test/test_noclass") #最终测试
    CLASSNAME_FILE = os.path.join(DATA_ROOT, "classname.txt")
    
    # 模型保存路径
    # CHECKPOINT_DIR = r"c:\Users\jangb\Desktop\contest_group\train_eff\checkpoints2"
    # CHECKPOINT_DIR = "/data2/jiangwb/eff/checkpoints"
    # CHECKPOINT_DIR = "eff/checkpoints_yc4"
    CHECKPOINT_DIR ='eff'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 预训练权重下载缓存路径
    # 默认为 ~/.cache/torch/hub/checkpoints
    # 可以修改为自定义路径，例如: "/data2/jiangwb/eff/pretrained_weights"
    PRETRAINED_WEIGHTS_DIR = 'eff/pretrained_weights'  # 相对路径（本地测试用）
    # PRETRAINED_WEIGHTS_DIR = "/data2/jiangwb/eff/pretrained_weights"  # 集群上用绝对路径
    
    # 训练参数
    BATCH_SIZE = 64
    NUM_EPOCHS = 34
    LEARNING_RATE = 0.85e-4
    WEIGHT_DECAY = 0.05
    NUM_WORKERS = 8
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 4
    
    # 图像参数
    IMAGE_SIZE = 224
    NUM_CLASSES = 17
    
    # 频率通道处理
    USE_FREQ_CHANNELS = False  # 是否使用频率通道(高频+低频)特征，如果为True输出5通道，为False输出3通道
    LOW_PASS_SIZE = 12         # 低频掩码大小(仅当USE_FREQ_CHANNELS=True时有效)
    
    # 设备
    DEVICE = "cuda"
    
    # 多GPU配置
    USE_MULTI_GPU = False           # 是否使用多GPU训练
    GPU_IDS = [0, 1]              # 使用的GPU索引（例：[0, 1] 表示使用第0和第1块GPU）
    
    # 随机种子
    SEED = 7
    
    # 验证集比例（从训练集分割）
    VAL_SPLIT = 0.2
    
    # 模型类型选择
    # EfficientNetV2: 'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l'
    # ConvNeXt V1: 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'
    # ConvNeXtV2: 'convnextv2_tiny', 'convnextv2_base', 'convnextv2_large', 'convnextv2_huge' (需要安装 timm: pip install timm)
    MODEL_TYPE = 'convnextv2_base'

    # ConvNeXtV2 结构改造：仅在 stage3 注入 CBAM
    USE_CBAM_STAGE3 = False
    CBAM_REDUCTION = 16
    CBAM_SPATIAL_KERNEL = 7
    
    # 是否对训练集进行下采样（每个类别最多使用200个样本）
    IF_UNDERAMPLE = False
    undersample_num = 250
    
    # 是否使用均衡BatchSampler（每个batch中各类别样本比例为1:1）
    # 当为True时，batch_size会自动调整为类别数的倍数
    IF_OVERSAMPLE = True
    
    # 集成学习配置
    # 要训练的模型数量
    NUM_ENSEMBLE_MODELS = 5
    # 是否在训练完成后自动运行集成预测
    AUTO_PREDICT_ENSEMBLE = True
    # 集成预测策略: 'average' (平均), 'voting' (投票), 'weighted' (加权)
    ENSEMBLE_STRATEGY = 'average'
    
    # 是否自动恢复训练（从最新的检查点继续）
    RESUME = False

    # 损失函数配置
    LOSS_TYPE = 'cross_entropy'          # 'cross_entropy' | 'focal'
    FOCAL_GAMMA = 1.5            # focal loss 聚焦参数
    LABEL_SMOOTHING = 0.1        # 标签平滑
    USE_CLASS_ALPHA = True     # 是否按类别频率自动计算 alpha 权重

    # 数据增强配置（batch级别）
    AUG_TYPE = 'cutmix_yolo'   # 'none' | 'mixup' | 'cutmix' | 'cutmix_yolo' | 'occamix' | 'both'(mixup/cutmix随机) | 'both_all'(四者随机)
    AUG_ALPHA = 2      # Beta分布参数：mixup推荐0.4，cutmix推荐1.0
    AUG_PROB = 0.5        # 每个batch执行增强的概率

    # 后置增强：在 mixup/cutmix/cutmix_yolo 之后再执行（仅 RGB 训练）
    POST_AUG_ENABLE = True
    POST_AUG_HFLIP_P = 0.5
    POST_AUG_ROTATE_DEGREES = 15
    POST_AUG_BRIGHTNESS = 0.1
    POST_AUG_CONTRAST = 0.1
    POST_AUG_SATURATION = 0.1
    POST_AUG_HUE = 0.1
    POST_AUG_SCALE_AREA_ADAPTIVE = True  # 是否按YOLO框面积自适应缩放区间（控制所有缩放）
    POST_AUG_SCALE_AREA_PROB = 0.5  # 自适应缩放触发概率
    POST_AUG_SCALE_AREA_SMALL_THRES = 0.1  # 小目标阈值（面积比）
    POST_AUG_SCALE_AREA_LARGE_THRES = 0.4  # 大目标阈值（面积比）
    POST_AUG_SCALE_JITTER_SMALL_BOX = 0.25  # 小目标仅放大: scale in [1.0, 1.0+jitter]
    POST_AUG_SCALE_JITTER_MID_BOX = 0.25  # 中等目标双向抖动: scale in [1.0-jitter, 1.0+jitter]
    POST_AUG_SCALE_JITTER_LARGE_BOX = 0.45  # 大目标仅缩小: scale in [1.0-jitter, 1.0]
    POST_AUG_NOISE_STD = 0       # 轻度高斯噪声标准差（像素域 0~1）
    POST_AUG_SP_NOISE_P = 0       # 轻量椒盐噪声像素概率（salt/pepper 各占一半）

    # YOLO 引导 CutMix（离线缓存）
    YOLO_CUTMIX_ENABLE = True
    YOLO_CUTMIX_CACHE_PATH = "eff/yolo_boxes_cache.json"
    YOLO_CUTMIX_KEY_MODE = 'relative_to_train_dir'  # 'relative_to_train_dir' | 'absolute'
    YOLO_CUTMIX_FALLBACK = 'skip'                   # 'skip' | 'random'
    YOLO_CUTMIX_MIN_BOX_AREA_RATIO = 0
    YOLO_CUTMIX_MAX_BOX_AREA_RATIO = 0.8
    YOLO_CUTMIX_CENTER_TOLERANCE_RATIO = 0.10
    YOLO_CUTMIX_DEBUG_LOG = False

    # YOLO-CutMix 配对策略：20% 随机 + 80% 面积阈值匹配（失败回退随机）
    YOLO_CUTMIX_PAIR_RANDOM_PROB = 0.2
    YOLO_CUTMIX_PAIR_AREA_RATIO_MIN = 0.33
    YOLO_CUTMIX_PAIR_AREA_RATIO_MAX = 3
    # 动态收紧目标窗口（训练后期收紧以提升稳定性）
    YOLO_CUTMIX_PAIR_AREA_RATIO_MIN_TARGET = 0.1
    YOLO_CUTMIX_PAIR_AREA_RATIO_MAX_TARGET = 2.00
    YOLO_CUTMIX_PAIR_SCHEDULE_START_RATIO = 0.50
    YOLO_CUTMIX_PAIR_SCHEDULE_END_RATIO = 0.7

    # 样本级路由：对不适配 YOLO-CutMix 的样本单独处理
    YOLO_CUTMIX_SAMPLE_ROUTING_ENABLE = False
    YOLO_CUTMIX_NON_ELIGIBLE_POLICY = 'none'       # 'mixup' | 'none'
    YOLO_CUTMIX_NON_ELIGIBLE_MIXUP_ALPHA = 0.4
    YOLO_CUTMIX_MIN_ELIGIBLE_RATIO = 0.0            # 预留：低于阈值可触发整批降级

    # OcCaMix 配置（仅当 AUG_TYPE='occamix' 或 'both_all' 时生效）
    OCCAMIX_N = 3                 # 减少混合区域，避免覆盖
    OCCAMIX_SEG_MIN = 40          # 更细粒度，贴合小种子
    OCCAMIX_SEG_MAX = 80          # 最大粒度，保证细节
    OCCAMIX_COMPACTNESS = 5.0     # 更贴合轮廓，减少跨区域

    # 错误样本保存配置
    SAVE_ERROR_SAMPLES = False  # 是否保存验证错误的样本图片
    ERROR_SAMPLES_DIR = "eff/checkpoints_base1/error_samples"  # 错误样本保存母文件夹

    # ====== 增强可视化保存（用于核验 CutMix / OcCaMix 是否生效）======
    SAVE_AUG_PREVIEW = False
    SAVE_AUG_PREVIEW_MAX_BATCHES = 2    # 每个 epoch 最多保存 1~3 个增强 batch
    SAVE_AUG_PREVIEW_MAX_SAMPLES = 6    # 每个 batch 最多保存几对图（原图/增强图）