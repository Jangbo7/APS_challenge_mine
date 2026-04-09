import os


class RoutedTestConfig:
    """Configuration for size-routed dual-model test inference."""

    ORIGINAL_TEST_DIR = os.path.join("data", "APS_dataset", "test", "test_noclass")
    PREPROCESSED_TEST_DIR = os.path.join("data", "APS_dataset_yolo224", "test_noclass")
    CLASSNAME_FILE = os.path.join("data", "APS_dataset", "classname.txt")

    YOLO_TEST_CACHE_PATH = "eff/yolo_boxes_cache_test.json"
    YOLO_KEY_MODE = "relative_to_train_dir"  # 'relative_to_train_dir' | 'absolute'

    RAW_MODEL_CHECKPOINT = os.path.join("eff", "best_loss-0873.pth")
    CROP224_MODEL_CHECKPOINT = os.path.join("eff", "latest.pth")

    RAW_MODEL_OVERRIDES = {
        "DATA_ROOT": os.path.join("data", "APS_dataset"),
        "TEST_DIR": ORIGINAL_TEST_DIR,
        "CLASSNAME_FILE": CLASSNAME_FILE,
        "IMAGE_SIZE": 224,
    }
    CROP224_MODEL_OVERRIDES = {
        "DATA_ROOT": os.path.join("data", "APS_dataset_yolo224"),
        "TEST_DIR": PREPROCESSED_TEST_DIR,
        "CLASSNAME_FILE": CLASSNAME_FILE,
        "IMAGE_SIZE": 224,
    }

    SMALL_AREA_THRES = 0.06
    LARGE_AREA_THRES = 0.25

    SMALL_POLICY = "crop224_only"
    MID_POLICY = "average"
    LARGE_POLICY = "raw_only"
    NO_BOX_POLICY = "raw_only"

    OUTPUT_FILE = os.path.join("eff", "result_routed.txt")
    DEVICE = "cuda"
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    PIN_MEMORY = True
