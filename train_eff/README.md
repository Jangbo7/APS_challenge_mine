# EfficientNetV2 训练测试框架

这是一个基于 PyTorch 和 EfficientNetV2 的图像分类训练测试框架，用于 APSNet 项目的植物种子分类任务。
目前预计尝试使用effcientnetv2作为基础分类模型。

## 目录结构

```
try/
├── config.py           # 配置文件
├── dataset.py          # 数据集加载（包含BalancedBatchSampler）
├── model.py            # EfficientNetV2 模型定义
├── train.py            # 训练脚本
├── train_ensemble.py   # 集成学习训练脚本
├── predict_ensemble.py # 集成预测脚本
├── test.py             # 测试脚本
├── utils.py            # 工具函数
├── metrics.py          # 类别级指标计算
└── checkpoints/        # 模型保存目录
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


## 高级功能

### 1. 训练集下采样 (IF_UNDERAMPLE)

当数据集存在严重的类别不均衡时，可以启用训练集下采样功能：

```python
# config.py
IF_UNDERAMPLE = True  # 启用下采样
```

**功能说明：**
- 对每个类别最多使用100个样本进行训练
- 验证集保持不变（不受下采样影响）
- 有助于快速原型验证和缓解类别不均衡

**使用场景：**
- 类别不均衡严重，需要快速验证模型架构
- 训练时间过长，需要减少训练数据量
- 测试不同超参数配置

### 2. 均衡Batch采样 (IF_OVERSAMPLE)

当需要确保每个batch中各类别样本比例为1:1时，可以启用均衡BatchSampler：

```python
# config.py
IF_OVERSAMPLE = True  # 启用均衡BatchSampler
```

**功能说明：**
- 每个batch中各类别样本数量相同（1:1比例）
- 自动调整batch size为类别数的倍数
- 最后一个batch样本不足时，随机选择部分类别填充

**工作原理：**
1. 计算每个batch中每个类别应包含的样本数 n
2. 正常情况下，每个batch包含所有类别，每个类别 n 个样本
3. 当最后一个batch样本不足时，随机选择部分类别来填充

**使用场景：**
- 类别不均衡严重，需要确保每个batch中各类别均衡
- 希望模型在每个迭代步骤都能看到均衡的类别分布
- 提升少数类的学习效果

### 3. 组合使用

可以同时启用下采样和均衡BatchSampler：

```python
# config.py
IF_UNDERAMPLE = True  # 下采样 + 均衡采样
IF_OVERSAMPLE = True
```

**效果：**
- 训练集：每个类别最多100个样本
- 每个batch：随机选择类别，保证batch size一致且类别均衡
- 验证集：保持原始分布不变

### 4. 类别级指标监控

训练过程中会自动计算并打印每个类别的详细指标：

```
Class-wise Metrics
================================================================================
Class                Support    Precision    Recall       F1-Score    
--------------------------------------------------------------------------------
Acalypha australis   20         0.9500       0.9200       0.9348      
Bassia scoparia      20         0.8800       0.8600       0.8696      
...
--------------------------------------------------------------------------------
Overall              160        0.9100       0.8938       0.9018      
================================================================================
```

**指标包括：**
- Precision（精确率）
- Recall（召回率）
- F1-Score（F1分数）
- Support（样本数）
- Confusion Matrix（混淆矩阵）

### 5. 增加两个频率通道
在config中通过use_freq_channel调控是否使用，low_pass_size调控低频掩码大小。




## 集成学习

### 配置参数

在 `config.py` 中配置集成学习相关参数：

```python
# 要训练的模型数量
NUM_ENSEMBLE_MODELS = 5

# 是否在训练完成后自动运行集成预测
AUTO_PREDICT_ENSEMBLE = False

# 集成预测策略: 'average' (平均), 'voting' (投票), 'weighted' (加权)
ENSEMBLE_STRATEGY = 'average'
```

### 训练多个模型

使用 `train_ensemble.py` 可以训练多个独立的模型用于集成：

```bash
python train_ensemble.py
```

**功能说明：**
- 自动训练多个模型（数量由 `NUM_ENSEMBLE_MODELS` 配置）
- 每个模型使用不同的随机种子
- 所有模型保存在独立的子目录中
- 训练完成后自动生成训练报告
- 如果 `AUTO_PREDICT_ENSEMBLE=True`，训练完成后自动运行集成预测

### 集成预测

使用 `predict_ensemble.py` 对测试集进行集成预测：

```bash
python predict_ensemble.py
```

**支持的集成策略：**

1. **average - 平均预测概率（推荐）**
   - 对所有模型的预测概率取平均
   - 通常效果最好

2. **voting - 投票**
   - 每个模型投票选择类别
   - 选择票数最多的类别

3. **weighted - 加权平均**
   - 根据模型验证集准确率加权
   - 准确率高的模型权重更大

**输出：**
- 生成 `ensemble_predictions.csv` 文件
- 包含文件名和预测类别

### 集成学习的优势

- **提升性能**：多个模型的集成通常比单个模型表现更好
- **降低过拟合风险**：不同模型的错误可以相互抵消
- **提高鲁棒性**：对数据扰动更加稳定

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

## 新增文件说明

### metrics.py
类别级指标计算模块，提供以下功能：

- `compute_class_metrics()`: 计算每个类别的Precision、Recall、F1-Score
- `print_class_metrics()`: 打印类别级指标表格
- `print_confusion_matrix()`: 打印混淆矩阵
- `get_class_statistics()`: 获取数据集类别分布统计
- `compute_class_accuracy()`: 计算每个类别的准确率

### dataset.py 新增功能

- `BalancedBatchSampler`: 均衡Batch采样器，确保每个batch中各类别比例为1:1
- `undersample_train_set()`: 训练集下采样函数
- `calculate_optimal_batch_size()`: 计算最优batch size
- `split_dataset_stratified()`: 分层分割数据集
