import os


class YoloPrecomputeConfig:
    """Configuration for offline YOLO cache generation and cache-based preprocessing."""

    # Input / output paths for YOLO box cache generation
    # TRAIN_DIR = os.path.join("data", "APS_dataset", "train")
    TRAIN_DIR = os.path.join("data", "APS_dataset", "test", "test_noclass")
    OUTPUT_CACHE_PATH = "eff/yolo_boxes_cache_test.json"

    # YOLO inference settings used by precompute_yolo_boxes.py
    MODEL = "yolo11m.pt"
    CONF = 0.055
    MAX_DET = 1
    PRIMARY_IMGSZ = 640
    IMGSZ = PRIMARY_IMGSZ
    ENABLE_SECOND_PASS = True
    SECOND_PASS_CONF = 0.005
    SECOND_PASS_IMGSZ = 1024
    DEVICE = "cuda"

    # Cache settings
    KEY_MODE = "relative_to_train_dir"  # 'relative_to_train_dir' | 'absolute'
    OVERWRITE = True

    # Preview settings
    YOLO_PREVIEW_ENABLE = True
    YOLO_PREVIEW_NUM = 2000
    YOLO_PREVIEW_BOX_TYPE = "both"  # 'with_boxes' | 'without_boxes' | 'both'
    YOLO_PREVIEW_OUTPUT_DIR = "eff/yolo_preview"
    YOLO_PREVIEW_BOX_COLOR = (0, 255, 0)
    YOLO_PREVIEW_BOX_THICKNESS = 2
    YOLO_PREVIEW_TEXT_COLOR = (0, 0, 255)

    # Cache-based 224 preprocessing settings
    PREPROCESS_INPUT_DIR = TRAIN_DIR
    PREPROCESS_OUTPUT_DIR = os.path.join("data", "APS_dataset_yolo224", "test_noclass")
    PREPROCESS_CACHE_PATH = OUTPUT_CACHE_PATH
    PREPROCESS_TARGET_SIZE = 224
    PREPROCESS_SMALL_AREA_THRES = 0.05
    PREPROCESS_LARGE_AREA_THRES = 0.307
    PREPROCESS_LINEAR_A = 1120.0
    PREPROCESS_LINEAR_B = 168.0
    PREPROCESS_NO_BOX_POLICY = "center_crop_224"
    PREPROCESS_OUT_OF_BOUNDS_POLICY = "shift_into_image"
    PREPROCESS_BOX_SELECT = "first_valid"
    PREPROCESS_FORCE_RESIZE_SAMPLES = [
        # "08iziczlb1YL.jpg",
        # "Acalypha australis/example.jpg",
        '1v1lfzXjKmbV.jpg',
        'i1i8KuQTzDmc.jpg',
        "iH6AUWvJhZhg.jpg",
        'HZnV4APdxHAG.jpg',
        'jrwRfDW14dty.jpg',
    ]
    PREPROCESS_OVERWRITE = True

    # Example absolute paths:
    # TRAIN_DIR = "/data2/jiangwb/eff/data/APS_dataset/train"
    # OUTPUT_CACHE_PATH = "/data2/jiangwb/eff/yolo_boxes_cache.json"
    # DEVICE = "cuda:0"
