import os


class Config2:
    DATA_ROOT_RAW = os.path.join("data", "APS_dataset")
    DATA_ROOT_DETAIL = os.path.join("data", "APS_dataset_yolo224")

    RAW_TRAIN_DIR = os.path.join(DATA_ROOT_RAW, "train")
    DETAIL_TRAIN_DIR = os.path.join(DATA_ROOT_DETAIL, "train")
    CLASSNAME_FILE = os.path.join(DATA_ROOT_RAW, "classname.txt")

    # "dual_view" or "single_view"
    TRAIN_MODE = "single_view"
    # used only when TRAIN_MODE == "single_view"
    # raw / detail
    SINGLE_VIEW_SOURCE = "detail"

    CHECKPOINT_DIR = "eff2/single_detail-f"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    PRETRAINED_WEIGHTS_DIR = "eff/pretrained_weights"

    BATCH_SIZE = 32
    NUM_EPOCHS = 28
    LEARNING_RATE = 0.82e-4
    WEIGHT_DECAY = 0.05
    NUM_WORKERS = 8
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 4
    PIN_MEMORY = True

    IMAGE_SIZE = 224
    NUM_CLASSES = 17
    DEVICE = "cuda"
    SEED = 17
    VAL_SPLIT = 0
    SAVE_LAST_K_EPOCHS = 4
    RESUME = False
    SELF_TRAIN_ENABLE = False
    SELF_TRAIN_EPOCHS = 8
    SELF_TRAIN_LEARNING_RATE = 0.25e-4
    SELF_TRAIN_INIT_CHECKPOINT = os.path.join("eff2", "single_detail-f", "latest.pth")
    SELF_TRAIN_SOURCE_DIR = (
        os.path.join(DATA_ROOT_RAW, "test", "test_noclass")
        if SINGLE_VIEW_SOURCE == "raw"
        else os.path.join(DATA_ROOT_DETAIL, "test_noclass")
    )
    SELF_TRAIN_YOLO_CACHE_PATH = "eff/yolo_boxes_cache_test.json"
    PSEUDO_CONF_THRESHOLD = 0.90
    PSEUDO_MARGIN_THRESHOLD = 0.25
    PSEUDO_USE_ENTROPY_FILTER = False
    PSEUDO_ENTROPY_THRESHOLD = 0.20
    PSEUDO_LOSS_WEIGHT = 0.2
    PSEUDO_MAX_SAMPLES_PER_CLASS = 0
    PSEUDO_BATCH_RATIO_LOG = 0.2
    # "dual_branch" 或 "single_backbone_6ch"
    MODEL_VARIANT = "single_backbone_6ch"
    MODEL_TYPE = "convnextv2_base"
    DROPOUT = 0.05
    FUSION_HIDDEN_DIM = 1024
    FUSION_DROPOUT = 0.2
    RAW_AUX_LOSS_WEIGHT = 0.3
    DETAIL_AUX_LOSS_WEIGHT = 0.3

    USE_FREQ_CHANNELS = False
    USE_CBAM_STAGE3 = False
    CBAM_REDUCTION = 16
    CBAM_SPATIAL_KERNEL = 7

    LOSS_TYPE = "cross_entropy"
    FOCAL_GAMMA = 1.5
    LABEL_SMOOTHING = 0.1
    USE_CLASS_ALPHA = True

    USE_MULTI_GPU = False
    GPU_IDS = [0, 1]

    DUALVIEW_ENABLE = True
    CUTMIX_ENABLE = True
    CUTMIX_PROB = 0.35
    CUTMIX_ALPHA = 1.4

    DEFECT_ENABLE = False
    DEFECT_PROB = 0.35
    DEFECT_N_TOP = 1
    DEFECT_SEG_MIN = 100
    DEFECT_SEG_MAX = 120
    DEFECT_COMPACTNESS = 1
    DEFECT_CAM_CHECKPOINT_PATH = "eff2/single_1/best_loss.pth"
    DEFECT_BORDER_WIDTH = 1
    DEFECT_BORDER_QUANTIZE = 16
    DEFECT_TARGET_EXPAND_RATIO = 0.0
    DEFECT_FILL_BLUR_SIGMA = 0.25

    DEFECT_VAL_ENABLE = False
    DEFECT_VAL_N_TOP = DEFECT_N_TOP
    DEFECT_VAL_SEG_MIN = DEFECT_SEG_MIN
    DEFECT_VAL_SEG_MAX = DEFECT_SEG_MAX
    DEFECT_VAL_COMPACTNESS = DEFECT_COMPACTNESS
    DEFECT_VAL_CAM_CHECKPOINT_PATH = DEFECT_CAM_CHECKPOINT_PATH
    DEFECT_VAL_BORDER_WIDTH = DEFECT_BORDER_WIDTH
    DEFECT_VAL_BORDER_QUANTIZE = DEFECT_BORDER_QUANTIZE
    DEFECT_VAL_TARGET_EXPAND_RATIO = DEFECT_TARGET_EXPAND_RATIO
    DEFECT_VAL_FILL_BLUR_SIGMA = DEFECT_FILL_BLUR_SIGMA

    YOLO_CACHE_PATH = "eff/yolo_boxes_cache.json"
    YOLO_KEY_MODE = "relative_to_train_dir"
    YOLO_MIN_BOX_AREA_RATIO = 0.0
    YOLO_MAX_BOX_AREA_RATIO = 0.8
    ENABLE_CENTER_SHIFT = True
    CENTER_TOLERANCE_RATIO = 0.0
    SECTOR_CENTER_JITTER_RATIO = 0.0

    PAIR_USE_AREA_MATCH = True
    PAIR_RANDOM_PROB = 0.1
    PAIR_AREA_RATIO_MIN = 0.5
    PAIR_AREA_RATIO_MAX = 2

    BG_AUG_ENABLE = True
    BG_AUG_PROB = 0.15
    BG_BLEED_INTO_BOX_RATIO = 0.1
    BG_BLACK_DOT_PROB = 0.0005
    BG_BLACK_DOT_SIZE_MIN = 2
    BG_BLACK_DOT_SIZE_MAX = 3
    BG_BLUR_SIGMA_MIN = 1.2
    BG_BLUR_SIGMA_MAX = 2.8
    BG_BRIGHTNESS = 0.45
    BG_CONTRAST = 0.55
    BG_SATURATION = 0.50
    BG_HUE = 0.3

    ROTATE_ENABLE = True
    ROTATE_PROB = 0.15
    ROTATE_DEGREES = 20.0
    ROTATE_FILL = 0.0
    FLIP_ENABLE = True
    FLIP_PROB = 0.15
    FLIP_VERTICAL = True

    SAVE_ERROR_SAMPLES = False
    ERROR_SAMPLES_DIR = os.path.join(CHECKPOINT_DIR, "error_samples")
