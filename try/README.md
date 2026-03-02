# EfficientNetV2 训练测试框架

这是一个基于 PyTorch 和 EfficientNetV2 的图像分类训练测试框架，用于 APSNet 项目的植物种子分类任务。

## 目录结构

```
try/
├── config.py      # 配置文件
├── dataset.py     # 数据集加载
├── model.py       # EfficientNetV2 模型定义
├── train.py       # 训练脚本
├── test.py        # 测试脚本
├── utils.py       # 工具函数
└── checkpoints/   # 模型保存目录
```

## 数据格式

数据目录结构：
```
data/APS_dataset/
├── train/                    # 训练集（会被分割为训练+验证）
│   ├── Acalypha australis/   # 类别1
│   ├── Bassia scoparia/      # 类别2
│   └── ...
├── val_noclass/              # 无标签测试集
└── classname.txt             # 类别名称列表
```

**重要说明：**
- 训练集会自动按 `VAL_SPLIT` 比例（默认20%）分层分割出验证集
- `val_noclass` 作为无标签的测试集，用于最终预测

## 使用方法

### 1. 训练模型

```bash
cd c:\Users\jangb\Desktop\contest\APSNet\try
python train.py
```

训练过程中会：
- 自动从训练集分割 20% 作为验证集（分层采样）
- 保存最佳模型到 `checkpoints/best.pth`
- 保存最新模型到 `checkpoints/latest.pth`
- 记录训练日志到 `checkpoints/train_YYYYMMDD_HHMMSS.log`

### 2. 测试模型

```bash
python test.py
```

测试脚本会：
- 加载最佳模型进行预测
- 对 `val_noclass` 中的无标签图片进行预测
- 生成预测结果 CSV 文件
- 支持交互式单张图片测试

### 3. 修改配置

编辑 `config.py` 文件可调整：
- `BATCH_SIZE`: 批次大小（默认32）
- `NUM_EPOCHS`: 训练轮数（默认50）
- `LEARNING_RATE`: 学习率（默认1e-4）
- `IMAGE_SIZE`: 输入图像尺寸（默认224）
- `DEVICE`: 训练设备（默认cuda）
- `VAL_SPLIT`: 验证集分割比例（默认0.2）

## 模型说明

默认使用 **EfficientNetV2-S** 模型，支持以下变体：
- `s`: EfficientNetV2-S (默认)
- `m`: EfficientNetV2-M
- `l`: EfficientNetV2-L

可在 `model.py` 中修改 `model_type` 参数切换模型。

## 依赖

- PyTorch >= 1.12
- torchvision >= 0.13
- tqdm
- pandas
- Pillow

## 类别信息

共17个类别：
1. Acalypha australis
2. Bassia scoparia
3. Cannabis sativa charred
4. Cannabis sativa soaked
5. Digitaria
6. Glycine max
7. Hordeum vulgare
8. Lespedeza bicolor
9. Melilotus suaveolens
10. Oryza sativa
11. Panicum miliaceum
12. Portulaca oleracea
13. Prunus persica
14. Setaria
15. Setaria italica
16. Sorghum bicolor
17. Triticum aestivum
