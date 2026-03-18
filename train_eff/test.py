import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

from config import Config
from dataset import get_test_dataloader, get_transforms
from model import build_model
from utils import load_checkpoint, accuracy, AverageMeter


@torch.no_grad()
def test(model, test_loader, device, class_names):
    """测试模型并生成预测结果"""
    model.eval()
    
    all_preds = []
    all_paths = []
    
    pbar = tqdm(test_loader, desc="[Testing]")
    for images, _, paths in pbar:
        images = images.to(device)
        
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_paths.extend(paths)
    
    results = []
    for path, pred in zip(all_paths, all_preds):
        results.append({
            'image': os.path.basename(path),
            'predicted_label': pred
        })
    
    return results


def save_results_to_txt(results, output_file):
    """
    保存预测结果为txt文件
    格式：每行 "图片名 预测标签"
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            line = f"{item['image']} {item['predicted_label']}\n"
            f.write(line)
    print(f"Results saved to: {output_file}")


def save_results_to_csv(results, class_names, output_file):
    """保存详细预测结果为CSV文件"""
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Detailed results saved to: {output_file}")


@torch.no_grad()
def validate(model, val_loader, device, class_names):
    """验证模型（有标签）"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    all_paths = []
    
    pbar = tqdm(val_loader, desc="[Validating]")
    for images, labels, paths in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        acc1 = accuracy(outputs, labels, topk=(1,))[0]
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_paths.extend(paths)
        
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{top1.avg:.2f}%'
        })
    
    print(f"\nValidation Results:")
    print(f"  Loss: {losses.avg:.4f}")
    print(f"  Accuracy: {top1.avg:.2f}%")
    
    return losses.avg, top1.avg, all_preds, all_labels, all_paths


@torch.no_grad()
def test_single_image(model, image_path, transform, device, class_names):
    """测试单张图片"""
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    output = model(image_tensor)
    prob = torch.softmax(output, dim=1)
    pred = output.argmax(dim=1).item()
    
    top5_prob, top5_idx = torch.topk(prob, k=min(5, len(class_names)), dim=1)
    top5_prob = top5_prob.squeeze().cpu().numpy()
    top5_idx = top5_idx.squeeze().cpu().numpy()
    
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Predicted class: {class_names[pred]} (confidence: {prob[0][pred]:.4f})")
    print("\nTop-5 predictions:")
    for i, (idx, p) in enumerate(zip(top5_idx, top5_prob)):
        print(f"  {i+1}. {class_names[idx]}: {p:.4f}")
    
    return pred, prob.squeeze().cpu().numpy()


def main():
    config = Config()
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    with open(config.CLASSNAME_FILE, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Number of classes: {len(class_names)}")
    
    print("Building model...")
    model = build_model(config)
    model = model.to(device)
    
    # checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best.pth')
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_acc.pth')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'latest.pth')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        load_checkpoint(model, None, checkpoint_path, device)
    else:
        print("Warning: No checkpoint found. Using random weights.")
    
    # 测试无标签测试集
    print(f"\n{'='*50}")
    print("Testing on unlabeled test set (val_noclass)...")
    print(f"{'='*50}")
    
    test_loader, _ = get_test_dataloader(config)
    
    results = test(model, test_loader, device, class_names)
    
    # 保存为txt文件（提交格式）
    txt_output_file = os.path.join(config.CHECKPOINT_DIR, 'result.txt')
    save_results_to_txt(results, txt_output_file)
    
    # 显示前10个预测结果
    print("\nFirst 10 predictions:")
    for i, item in enumerate(results[:10]):
        print(f"  {item['image']} {item['predicted_label']}")
    
    # 交互式测试单张图片
    print("\n" + "="*50)
    print("Interactive testing mode")
    print("Enter image path to test (or 'q' to quit):")
    
    test_transform = get_transforms(is_train=False, image_size=config.IMAGE_SIZE)
    
    while True:
        user_input = input("> ").strip()
        
        if user_input.lower() == 'q':
            break
        
        if not os.path.exists(user_input):
            print(f"Error: File not found: {user_input}")
            continue
        
        try:
            test_single_image(model, user_input, test_transform, device, class_names)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
