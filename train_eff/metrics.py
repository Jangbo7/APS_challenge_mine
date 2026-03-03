import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def compute_class_metrics(all_preds, all_labels, num_classes, class_names=None):
    """
    计算每个类别的详细指标
    
    Args:
        all_preds: 预测标签列表
        all_labels: 真实标签列表
        num_classes: 类别数量
        class_names: 类别名称列表
    
    Returns:
        metrics_dict: 包含各类指标的字典
    """
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算整体指标
    overall_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    overall_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    overall_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # 计算每个类别的指标
    class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    # 统计每个类别的样本数
    class_counts = np.bincount(all_labels, minlength=num_classes)
    
    # 构建指标字典
    metrics_dict = {
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        },
        'per_class': {
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1,
            'support': class_counts
        },
        'confusion_matrix': cm
    }
    
    return metrics_dict


def print_class_metrics(metrics_dict, class_names=None):
    """
    打印每个类别的指标
    
    Args:
        metrics_dict: compute_class_metrics 返回的指标字典
        class_names: 类别名称列表
    """
    print("\n" + "="*80)
    print("Class-wise Metrics")
    print("="*80)
    
    num_classes = len(metrics_dict['per_class']['precision'])
    
    # 表头
    header = f"{'Class':<20} {'Support':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}"
    print(header)
    print("-"*80)
    
    # 每个类别的指标
    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        support = int(metrics_dict['per_class']['support'][i])
        precision = metrics_dict['per_class']['precision'][i]
        recall = metrics_dict['per_class']['recall'][i]
        f1 = metrics_dict['per_class']['f1'][i]
        
        row = f"{class_name:<20} {support:<10} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}"
        print(row)
    
    # 整体指标
    print("-"*80)
    overall = metrics_dict['overall']
    overall_row = f"{'Overall':<20} {sum(metrics_dict['per_class']['support']):<10} {overall['precision']:<12.4f} {overall['recall']:<12.4f} {overall['f1']:<12.4f}"
    print(overall_row)
    print("="*80 + "\n")


def print_confusion_matrix(cm, class_names=None):
    """
    打印混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
    """
    print("Confusion Matrix:")
    print("-"*60)
    
    num_classes = cm.shape[0]
    
    # 打印表头
    if class_names:
        header = "Predicted -> " + " ".join([f"{name[:8]:<8}" for name in class_names])
    else:
        header = "Predicted -> " + " ".join([f"C{i:<7}" for i in range(num_classes)])
    print(header)
    
    # 打印每一行
    for i in range(num_classes):
        if class_names:
            row_label = f"Actual {class_names[i][:8]:<8}"
        else:
            row_label = f"Actual C{i:<8}"
        
        row = row_label + " " + " ".join([f"{cm[i,j]:<8}" for j in range(num_classes)])
        print(row)
    
    print("-"*60 + "\n")


def get_class_statistics(all_labels, class_names=None):
    """
    获取数据集的类别统计信息
    
    Args:
        all_labels: 真实标签列表
        class_names: 类别名称列表
    
    Returns:
        class_counts: 每个类别的样本数
    """
    all_labels = np.array(all_labels)
    class_counts = np.bincount(all_labels)
    
    print("\n" + "="*60)
    print("Dataset Class Distribution")
    print("="*60)
    
    total_samples = len(all_labels)
    
    for i, count in enumerate(class_counts):
        if count > 0:
            class_name = class_names[i] if class_names else f"Class {i}"
            percentage = (count / total_samples) * 100
            print(f"{class_name:<20} {count:<8} samples ({percentage:.2f}%)")
    
    print("-"*60)
    print(f"Total: {total_samples} samples")
    print("="*60 + "\n")
    
    return class_counts


def compute_class_accuracy(all_preds, all_labels, num_classes):
    """
    计算每个类别的准确率
    
    Args:
        all_preds: 预测标签列表
        all_labels: 真实标签列表
        num_classes: 类别数量
    
    Returns:
        class_acc: 每个类别的准确率数组
    """
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    class_acc = np.zeros(num_classes)
    
    for i in range(num_classes):
        mask = (all_labels == i)
        if mask.sum() > 0:
            class_acc[i] = (all_preds[mask] == all_labels[mask]).mean()
    
    return class_acc


def print_class_accuracy(class_acc, class_names=None):
    """
    打印每个类别的准确率
    
    Args:
        class_acc: 每个类别的准确率数组
        class_names: 类别名称列表
    """
    print("\n" + "="*60)
    print("Class-wise Accuracy")
    print("="*60)
    
    for i, acc in enumerate(class_acc):
        class_name = class_names[i] if class_names else f"Class {i}"
        print(f"{class_name:<20} {acc*100:.2f}%")
    
    print("="*60 + "\n")
