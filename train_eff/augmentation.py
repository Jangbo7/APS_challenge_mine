import torch
import numpy as np
import random

try:
    from skimage import segmentation
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def mixup_data(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4):
    """
    MixUp 数据增强
    
    Args:
        images: [B, C, H, W]
        labels: [B] 类别索引
        alpha:  Beta分布参数，越大混合越均匀，推荐 0.2~0.8
    
    Returns:
        mixed_images: [B, C, H, W]
        labels_a:     原始标签
        labels_b:     混合标签
        lam:          混合比例（标量）
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    B = images.size(0)
    rand_idx = torch.randperm(B, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[rand_idx]
    labels_a = labels
    labels_b = labels[rand_idx]

    return mixed_images, labels_a, labels_b, lam


def cutmix_data(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    """
    CutMix 数据增强
    
    Args:
        images: [B, C, H, W]
        labels: [B] 类别索引
        alpha:  Beta分布参数，推荐 0.5~1.5
    
    Returns:
        mixed_images: [B, C, H, W]
        labels_a:     原始标签
        labels_b:     混合标签
        lam:          原始图像保留的面积比例
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    B, C, H, W = images.shape
    rand_idx = torch.randperm(B, device=images.device)

    # 生成随机裁剪框
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]

    # 用实际面积修正 lam
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)

    labels_a = labels
    labels_b = labels[rand_idx]

    return mixed_images, labels_a, labels_b, lam


def mixed_criterion(criterion, pred, labels_a, labels_b, lam):
    """
    混合损失计算：lam * loss(pred, a) + (1-lam) * loss(pred, b)
    
    注意：criterion 需要支持逐样本输出（reduction='none'）
          或者直接用两次 mean 加权
    """
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


def occamix_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    model,
    n_top: int = 6,
    n_seg_max: int = 50,
    n_seg_min: int = 20,
    compactness: float = 10.0,
    lam_beta: float = 1.5,
    mask_only_ratio: float = 0.0,
    mask_background: str = 'zero',
    mask_only_topk_superpixels_per_block: int = 4,
):
    """
    OcCaMix 数据增强（基于 attentive superpixel 的像素级混合）

    Args:
        images: [B, C, H, W]
        labels: [B]
        model:  需要实现 get_spatial_feature_map / get_classifier_weight
        n_top:  每张图选取 top-N attention 网格
        n_seg_max: SLIC 最大分段数
        n_seg_min: SLIC 最小分段数
        compactness: SLIC 紧致度
        lam_beta: lam校正系数，lam = lam_beta * mixed_area_ratio
        mask_only_ratio: 在 occamix 样本中使用“仅保留注意力超像素输入”的比例
        mask_background: mask-only 背景填充策略，当前支持 'zero'
        mask_only_topk_superpixels_per_block: mask-only 样本中，每个注意力块选择的超像素数量

    Returns:
        mixed_images: [B, C, H, W]
        labels_a:     原始标签
        labels_b:     置乱标签
        lam_batch:    [B] 每个样本“混入图标签权重”
        mask_only_flags: [B] 是否为 mask-only 样本
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("OcCaMix requires scikit-image. Install with: pip install scikit-image")

    bsz, _, h_img, w_img = images.shape
    rand_idx = torch.randperm(bsz, device=images.device)

    labels_a = labels
    labels_b = labels[rand_idx]
    mixed_images = images.clone()

    model_ref = model.module if hasattr(model, 'module') else model
    was_training = model.training

    model.eval()
    try:
        with torch.no_grad():
            feat_map = model_ref.get_spatial_feature_map(images[rand_idx])
            cls_weight = model_ref.get_classifier_weight().detach()

            if feat_map.ndim != 4:
                raise RuntimeError(f"OcCaMix expects 4D feature map [B,C,H,W], got {tuple(feat_map.shape)}")

            # CAM: [B, num_classes, Hf, Wf]
            cam_all = torch.einsum('kc,bchw->bkhw', cls_weight.to(feat_map.device), feat_map)
            cam_all = torch.relu(cam_all)

            bsz_f, _, h_fmap, w_fmap = cam_all.shape
            eval_train_map = cam_all.amax(dim=1).view(bsz_f, -1)  # [B, Hf*Wf]
    finally:
        if was_training:
            model.train()

    n_top = max(1, int(n_top))
    n_top = min(n_top, eval_train_map.size(1))

    _, map_topn_idx = torch.topk(eval_train_map, n_top, dim=1, largest=True)
    map_topn_row = (map_topn_idx // w_fmap).cpu().numpy()
    map_topn_col = (map_topn_idx % w_fmap).cpu().numpy()

    mask_only_ratio = float(max(0.0, min(1.0, mask_only_ratio)))
    lam_beta = float(max(0.0, lam_beta))
    mask_only_topk_superpixels_per_block = max(1, int(mask_only_topk_superpixels_per_block))
    lam_batch = []
    mask_only_flags = []
    for i in range(bsz):
        use_mask_only = (np.random.rand() < mask_only_ratio)
        per_block_topk = mask_only_topk_superpixels_per_block if use_mask_only else 1

        img_seg = images[rand_idx[i]].detach().permute(1, 2, 0).cpu().numpy()  # H,W,C
        n_seg = random.randint(int(n_seg_min), int(n_seg_max))
        segments_img_map = segmentation.slic(
            img_seg,
            n_segments=n_seg,
            compactness=float(compactness),
            start_label=0,
            channel_axis=-1,
        )

        selected_mask = np.zeros((h_img, w_img), dtype=bool)

        for k in range(n_top):
            row = int(map_topn_row[i][k])
            col = int(map_topn_col[i][k])

            y1 = int(row * h_img / h_fmap)
            y2 = int((row + 1) * h_img / h_fmap)
            x1 = int(col * w_img / w_fmap)
            x2 = int((col + 1) * w_img / w_fmap)

            y1 = max(0, min(y1, h_img - 1))
            y2 = max(y1 + 1, min(y2, h_img))
            x1 = max(0, min(x1, w_img - 1))
            x2 = max(x1 + 1, min(x2, w_img))

            atten_mask = np.zeros((h_img, w_img), dtype=bool)
            atten_mask[y1:y2, x1:x2] = True

            seg_labels = np.unique(segments_img_map[atten_mask])
            if seg_labels.size == 0:
                continue

            overlap_candidates = []

            for seg_label in seg_labels:
                superpixel_mask = segments_img_map == seg_label
                superpixel_area = superpixel_mask.sum()
                if superpixel_area == 0:
                    continue

                overlap = np.logical_and(atten_mask, superpixel_mask).sum()
                overlap_pct = overlap / superpixel_area

                overlap_candidates.append((overlap_pct, superpixel_mask))

            if overlap_candidates:
                overlap_candidates.sort(key=lambda x: x[0], reverse=True)
                topk_candidates = overlap_candidates[:per_block_topk]
                for _, chosen_superpixel_mask in topk_candidates:
                    selected_mask |= chosen_superpixel_mask

        mix_pixel_count = int(selected_mask.sum())
        mask_only_flags.append(use_mask_only)

        if mix_pixel_count > 0:
            selected_mask_t = torch.from_numpy(selected_mask).to(images.device)
            if use_mask_only:
                if mask_background != 'zero':
                    raise ValueError(f"Unsupported mask_background: {mask_background}. Only 'zero' is supported now.")
                masked_img = torch.zeros_like(mixed_images[i])
                masked_img[:, selected_mask_t] = images[i, :, selected_mask_t]
                mixed_images[i] = masked_img
                lam_batch.append(1.0)
            else:
                mixed_images[i, :, selected_mask_t] = images[rand_idx[i], :, selected_mask_t]
                mixed_area_ratio = mix_pixel_count / float(h_img * w_img)
                lam = lam_beta * mixed_area_ratio
                lam = float(max(0.0, min(1.0, lam)))
                lam_batch.append(lam)
        else:
            if use_mask_only:
                lam_batch.append(1.0)
            else:
                mixed_area_ratio = 0.0
                lam = lam_beta * mixed_area_ratio
                lam = float(max(0.0, min(1.0, lam)))
                lam_batch.append(lam)

    lam_batch = torch.tensor(lam_batch, dtype=images.dtype, device=images.device)
    mask_only_flags = torch.tensor(mask_only_flags, dtype=torch.bool, device=images.device)
    return mixed_images, labels_a, labels_b, lam_batch, mask_only_flags