import os


class YoloPrecomputeConfig:
    """YOLO 离线预生成缓存配置"""

    # 输入输出路径
    TRAIN_DIR = os.path.join( "data", "APS_dataset", "train")
    OUTPUT_CACHE_PATH = "eff/yolo_boxes_cache.json"

    # YOLO 模型配置
    MODEL = 'yolo11n.pt'          # YOLO 模型：yolo11n/yolo11s/yolo11m 等
    CONF = 0.005                  # 置信度阈值
    MAX_DET = 1                  # 每张图最多检测框数
    DEVICE = 'cuda'               # 推理设备：cuda/cpu

    # 缓存配置
    KEY_MODE = 'relative_to_train_dir'  # 缓存键模式：'relative_to_train_dir' | 'absolute'
    OVERWRITE = True             # 是否覆盖已有缓存

    # 预览配置（可视化检测结果）
    YOLO_PREVIEW_ENABLE = False                     # 是否保存预览图
    YOLO_PREVIEW_NUM = 100                           # 预览图片数量
    YOLO_PREVIEW_BOX_TYPE = 'both'            # 'with_boxes' | 'without_boxes' | 'both'
    YOLO_PREVIEW_OUTPUT_DIR = 'eff/yolo_preview'    # 预览图输出目录
    YOLO_PREVIEW_BOX_COLOR = (0, 255, 0)            # 框颜色 (BGR)：绿色
    YOLO_PREVIEW_BOX_THICKNESS = 2                  # 框线宽度
    YOLO_PREVIEW_TEXT_COLOR = (0, 0, 255)           # 文字颜色 (BGR)：红色

    # ====== 或者也可以使用绝对路径的示例 ======
    # TRAIN_DIR = "/data2/jiangwb/eff/data/APS_dataset/train"
    # OUTPUT_CACHE_PATH = "/data2/jiangwb/eff/yolo_boxes_cache.json"
    # DEVICE = 'cuda:0'  # 指定 GPU 编号
