import os


class DualViewTestConfig:
    TRAIN_MODE = "single_view"
    SINGLE_VIEW_SOURCE = "detail"

    RAW_TEST_DIR = os.path.join("data", "APS_dataset", "test", "test_noclass")
    DETAIL_TEST_DIR = os.path.join("data", "APS_dataset_yolo224", "test_noclass")
    CLASSNAME_FILE = os.path.join("data", "APS_dataset", "classname.txt")

    CHECKPOINT_PATH = os.path.join("eff/eff2/p_d", "latest.pth")
    OUTPUT_FILE = os.path.join("eff/eff2/detail", "result.txt")
    OUTPUT_CSV = os.path.join("eff/eff2/detail", "result.csv")

    DEVICE = "cuda"
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 4
    PIN_MEMORY = True

    IMAGE_SIZE = 224
    NUM_CLASSES = 17
    # "dual_branch" 或 "single_backbone_6ch"
    MODEL_VARIANT = "single_backbone_6ch"
    MODEL_TYPE = "convnextv2_base"
    FUSION_HIDDEN_DIM = 1024
    FUSION_DROPOUT = 0.2

    PRETRAINED_WEIGHTS_DIR = "eff/pretrained_weights"
    USE_FREQ_CHANNELS = False
    USE_CBAM_STAGE3 = False
    CBAM_REDUCTION = 16
    CBAM_SPATIAL_KERNEL = 7


class RoutedTrain2TestConfig:
    RAW_TEST_DIR = os.path.join("data", "APS_dataset", "test", "test_noclass")
    DETAIL_TEST_DIR = os.path.join("data", "APS_dataset_yolo224",  "test_noclass")
    CLASSNAME_FILE = os.path.join("data", "APS_dataset", "classname.txt")

    YOLO_TEST_CACHE_PATH = "eff/yolo_boxes_cache_test.json"
    YOLO_KEY_MODE = "relative_to_train_dir"  # 'relative_to_train_dir' | 'absolute'

    RAW_CHECKPOINT_PATH = os.path.join("eff", "eff2", "raw", "latest.pth")
    DETAIL_CHECKPOINT_PATH = os.path.join("eff", "eff2", "detail", "latest.pth")

    RAW_MODEL_TYPE = "convnextv2_base"
    DETAIL_MODEL_TYPE = "convnextv2_base"
    RAW_IMAGE_SIZE = 224
    DETAIL_IMAGE_SIZE = 224
    NUM_CLASSES = 17
    PRETRAINED_WEIGHTS_DIR = "eff/pretrained_weights"
    USE_FREQ_CHANNELS = False
    USE_CBAM_STAGE3 = False
    CBAM_REDUCTION = 16
    CBAM_SPATIAL_KERNEL = 7
    DROPOUT = 0.05

    SMALL_AREA_THRES = 0.06
    LARGE_AREA_THRES = 0.25

    SMALL_POLICY = "crop224_only"
    MID_POLICY = "average"
    LARGE_POLICY = "raw_only"
    NO_BOX_POLICY = "raw_only"

    OUTPUT_FILE = os.path.join("eff", "eff2", "routed_result.txt")
    OUTPUT_CSV = os.path.join("eff", "eff2", "routed_result.csv")

    DEVICE = "cuda"
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    PIN_MEMORY = True
