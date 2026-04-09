import torch
import numpy as np
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Dict, List, Optional, Sequence, Tuple
from yolo_cutmix import YoloCutMixHelper

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


def cutmix_data_yolo(
    images: torch.Tensor,
    labels: torch.Tensor,
    img_paths,
    yolo_helper: YoloCutMixHelper,
    alpha: float = 1.0,
):
    """
    YOLO 引导 CutMix（薄包装）：核心逻辑在 yolo_cutmix.py 中。

    Returns:
        mixed_images, labels_a, labels_b, lam_batch, stats, applied_mask, rand_idx
    """
    if yolo_helper is None:
        raise ValueError("cutmix_data_yolo requires a valid YoloCutMixHelper")

    return yolo_helper.apply(images=images, labels=labels, img_paths=img_paths, alpha=alpha)


def mixed_criterion(criterion, pred, labels_a, labels_b, lam):
    """
    混合损失计算：lam * loss(pred, a) + (1-lam) * loss(pred, b)
    
    注意：criterion 需要支持逐样本输出（reduction='none'）
          或者直接用两次 mean 加权
    """
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


def generate_occamix_masks(
    images: torch.Tensor,
    model,
    n_top: int = 6,
    n_seg_max: int = 50,
    n_seg_min: int = 20,
    compactness: float = 10.0,
    source_indices: Optional[torch.Tensor] = None,
):
    if not SKIMAGE_AVAILABLE:
        raise ImportError("OcCaMix requires scikit-image. Install with: pip install scikit-image")

    bsz, _, h_img, w_img = images.shape
    if source_indices is None:
        source_indices = torch.arange(bsz, device=images.device)
    if source_indices.numel() != bsz:
        raise ValueError("source_indices must have the same batch size as images")

    model_ref = model.module if hasattr(model, 'module') else model
    was_training = model.training

    model.eval()
    with torch.no_grad():
        feat_map = model_ref.get_spatial_feature_map(images[source_indices])
        cls_weight = model_ref.get_classifier_weight().detach()

        if feat_map.ndim != 4:
            raise RuntimeError(f"OcCaMix expects 4D feature map [B,C,H,W], got {tuple(feat_map.shape)}")

        cam_all = torch.einsum('kc,bchw->bkhw', cls_weight.to(feat_map.device), feat_map)
        cam_all = torch.relu(cam_all)

        bsz_f, _, h_fmap, w_fmap = cam_all.shape
        eval_train_map = cam_all.amax(dim=1).view(bsz_f, -1)

    if was_training:
        model.train()

    n_top = max(1, int(n_top))
    n_top = min(n_top, eval_train_map.size(1))

    _, map_topn_idx = torch.topk(eval_train_map, n_top, dim=1, largest=True)
    map_topn_row = (map_topn_idx // w_fmap).cpu().numpy()
    map_topn_col = (map_topn_idx % w_fmap).cpu().numpy()

    mask_batch: List[torch.Tensor] = []
    lam_batch = []
    for i in range(bsz):
        img_seg = images[source_indices[i]].detach().permute(1, 2, 0).cpu().numpy()
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

            best_overlap = -1.0
            best_superpixel_mask = None

            for seg_label in seg_labels:
                superpixel_mask = segments_img_map == seg_label
                superpixel_area = superpixel_mask.sum()
                if superpixel_area == 0:
                    continue

                overlap = np.logical_and(atten_mask, superpixel_mask).sum()
                overlap_pct = overlap / superpixel_area

                if overlap_pct > best_overlap:
                    best_overlap = overlap_pct
                    best_superpixel_mask = superpixel_mask

            if best_superpixel_mask is not None:
                selected_mask |= best_superpixel_mask

        mix_pixel_count = int(selected_mask.sum())
        mask_batch.append(torch.from_numpy(selected_mask))
        lam_batch.append(1.0 - (mix_pixel_count / float(h_img * w_img)))

    masks = torch.stack(mask_batch, dim=0).to(device=images.device, dtype=torch.bool)
    lam = torch.tensor(lam_batch, dtype=images.dtype, device=images.device)
    return masks, lam


def lookup_valid_boxes(helper: YoloCutMixHelper, img_path: str, h: int, w: int) -> List[Tuple[int, int, int, int]]:
    entry = helper._lookup_entry(img_path)
    boxes_raw, orig_size = helper._parse_entry(entry)
    return helper._build_valid_boxes(boxes_raw, orig_size, h=h, w=w)


def maybe_center_shift_image(
    image: torch.Tensor,
    helper: YoloCutMixHelper,
    img_path: str,
    rng,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    _, h, w = image.shape
    valid_boxes = lookup_valid_boxes(helper, img_path, h=h, w=w)
    info: Dict[str, object] = {
        "box_count": len(valid_boxes),
        "shift_dx": 0,
        "shift_dy": 0,
        "foreground_box": None,
    }
    if not valid_boxes:
        return image, info

    chosen_box = valid_boxes[rng.randrange(len(valid_boxes))]
    shifted_box = chosen_box
    shifted_image = image
    if helper.enable_recenter_shift:
        dx, dy = helper._compute_recenter_shift(chosen_box, h=h, w=w)
        info["shift_dx"] = int(dx)
        info["shift_dy"] = int(dy)
        if dx != 0 or dy != 0:
            shifted_image = helper._translate_image(image, dx, dy)
            shifted_box = helper._translate_box(chosen_box, dx, dy, h=h, w=w)
    if shifted_box is not None:
        info["foreground_box"] = [int(v) for v in shifted_box]
    return shifted_image, info


def build_background_mask(
    h: int,
    w: int,
    foreground_box: Sequence[int],
    bleed_ratio: float,
    rng,
    device: torch.device,
) -> torch.Tensor:
    x1, y1, x2, y2 = [int(v) for v in foreground_box[:4]]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    background_mask = torch.ones((h, w), dtype=torch.bool, device=device)
    background_mask[y1:y2, x1:x2] = False

    if bleed_ratio <= 0:
        return background_mask

    box_h = max(1, y2 - y1)
    box_w = max(1, x2 - x1)
    bleed_x = int(round(box_w * bleed_ratio))
    bleed_y = int(round(box_h * bleed_ratio))
    if bleed_x <= 0 and bleed_y <= 0:
        return background_mask

    yy = torch.arange(box_h, device=device, dtype=torch.float32).view(box_h, 1).expand(box_h, box_w)
    xx = torch.arange(box_w, device=device, dtype=torch.float32).view(1, box_w).expand(box_h, box_w)

    prob_map = torch.zeros((box_h, box_w), dtype=torch.float32, device=device)
    if bleed_x > 0:
        left_prob = (1.0 - xx / float(max(bleed_x, 1))).clamp(min=0.0, max=1.0)
        right_prob = (1.0 - (box_w - 1 - xx) / float(max(bleed_x, 1))).clamp(min=0.0, max=1.0)
        prob_map = torch.maximum(prob_map, left_prob)
        prob_map = torch.maximum(prob_map, right_prob)
    if bleed_y > 0:
        top_prob = (1.0 - yy / float(max(bleed_y, 1))).clamp(min=0.0, max=1.0)
        bottom_prob = (1.0 - (box_h - 1 - yy) / float(max(bleed_y, 1))).clamp(min=0.0, max=1.0)
        prob_map = torch.maximum(prob_map, top_prob)
        prob_map = torch.maximum(prob_map, bottom_prob)

    if float(prob_map.max().item()) <= 0.0:
        return background_mask

    rs = np.random.RandomState(rng.randrange(0, 2**31 - 1))
    random_map = torch.from_numpy(rs.rand(box_h, box_w)).to(device=device, dtype=torch.float32)
    intrusion_mask = random_map < (0.55 * prob_map)
    background_mask[y1:y2, x1:x2] = intrusion_mask
    return background_mask


def apply_black_mosaic_dots(
    image: torch.Tensor,
    background_mask: torch.Tensor,
    rng,
    dot_prob: float,
    dot_size_min: int,
    dot_size_max: int,
) -> torch.Tensor:
    if dot_prob <= 0:
        return image

    coords = background_mask.nonzero(as_tuple=False)
    if coords.numel() == 0:
        return image

    coords_cpu = coords.detach().cpu()
    num_dots = int(round(coords_cpu.shape[0] * dot_prob))
    if num_dots <= 0:
        return image
    num_dots = min(num_dots, 1200)

    dot_size_min = max(1, int(dot_size_min))
    dot_size_max = max(dot_size_min, int(dot_size_max))
    output = image.clone()
    for _ in range(num_dots):
        y, x = coords_cpu[rng.randrange(coords_cpu.shape[0])].tolist()
        size = rng.randint(dot_size_min, dot_size_max)
        half = size // 2
        y1 = max(0, y - half)
        y2 = min(background_mask.shape[0], y1 + size)
        x1 = max(0, x - half)
        x2 = min(background_mask.shape[1], x1 + size)
        submask = background_mask[y1:y2, x1:x2]
        output[:, y1:y2, x1:x2] = torch.where(
            submask.unsqueeze(0),
            torch.zeros_like(output[:, y1:y2, x1:x2]),
            output[:, y1:y2, x1:x2],
        )
    return output


def apply_background_corruption(
    image: torch.Tensor,
    foreground_box: Optional[Sequence[int]],
    rng,
    bleed_ratio: float,
    black_dot_prob: float,
    black_dot_size_min: int,
    black_dot_size_max: int,
    blur_sigma_min: float,
    blur_sigma_max: float,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    info: Dict[str, object] = {
        "bg_aug_applied": False,
        "bg_mask_ratio": 0.0,
        "bg_preserve_box": None,
        "background_mask": None,
    }
    if foreground_box is None:
        return image, info

    _, h, w = image.shape
    background_mask = build_background_mask(h, w, foreground_box, bleed_ratio, rng, image.device)
    bg_ratio = float(background_mask.float().mean().item())
    if bg_ratio <= 0.0:
        return image, info

    corrupted = image.clone()
    if blur_sigma_max > 0:
        sigma_low = max(0.0, min(blur_sigma_min, blur_sigma_max))
        sigma_high = max(sigma_low, blur_sigma_max)
        sigma = rng.uniform(sigma_low, sigma_high)
        kernel_size = max(3, int(round(sigma * 4)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        corrupted = TF.gaussian_blur(corrupted, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    if brightness > 0:
        corrupted = TF.adjust_brightness(corrupted, 1.0 + rng.uniform(-brightness, brightness))
    if contrast > 0:
        corrupted = TF.adjust_contrast(corrupted, 1.0 + rng.uniform(-contrast, contrast))
    if saturation > 0:
        corrupted = TF.adjust_saturation(corrupted, 1.0 + rng.uniform(-saturation, saturation))
    if hue > 0:
        corrupted = TF.adjust_hue(corrupted, rng.uniform(-hue, hue))

    corrupted = apply_black_mosaic_dots(
        corrupted,
        background_mask,
        rng,
        dot_prob=black_dot_prob,
        dot_size_min=black_dot_size_min,
        dot_size_max=black_dot_size_max,
    )

    mask3 = background_mask.unsqueeze(0).expand_as(image)
    mixed = torch.where(mask3, corrupted, image)
    info["bg_aug_applied"] = True
    info["bg_mask_ratio"] = bg_ratio
    info["bg_preserve_box"] = [int(v) for v in foreground_box[:4]]
    info["background_mask"] = background_mask
    return mixed, info


def _box_to_mask(
    foreground_box: Sequence[int],
    h: int,
    w: int,
    device: torch.device,
) -> torch.Tensor:
    x1, y1, x2, y2 = [int(v) for v in foreground_box[:4]]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    mask[y1:y2, x1:x2] = True
    return mask


def _expand_mask(mask: torch.Tensor, expand_ratio: float) -> torch.Tensor:
    if expand_ratio <= 0:
        return mask
    h, w = mask.shape
    radius = int(round(float(min(h, w)) * expand_ratio))
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    expanded = F.max_pool2d(
        mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )
    return expanded.squeeze(0).squeeze(0) > 0


def fill_mask_from_background(
    image: torch.Tensor,
    target_mask: torch.Tensor,
    background_mask: torch.Tensor,
    rng,
    fill_blur_sigma: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    info = {
        "target_ratio": float(target_mask.float().mean().item()),
        "background_ratio": float(background_mask.float().mean().item()),
        "applied": 0.0,
    }
    bg_coords = background_mask.nonzero(as_tuple=False)
    target_coords = target_mask.nonzero(as_tuple=False)
    if bg_coords.numel() == 0 or target_coords.numel() == 0:
        return image, info

    output = image.clone()
    rand_ids = torch.randint(
        low=0,
        high=bg_coords.shape[0],
        size=(target_coords.shape[0],),
        device=image.device,
    )
    sampled_coords = bg_coords[rand_ids]
    output[:, target_coords[:, 0], target_coords[:, 1]] = image[:, sampled_coords[:, 0], sampled_coords[:, 1]]

    if fill_blur_sigma > 0:
        sigma = float(fill_blur_sigma)
        kernel_size = max(3, int(round(sigma * 4)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = TF.gaussian_blur(output, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        output = torch.where(target_mask.unsqueeze(0), blurred, output)

    info["applied"] = 1.0
    return output, info


def occamix_bgfill_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    img_paths,
    model,
    yolo_helper: YoloCutMixHelper,
    n_top: int = 6,
    n_seg_max: int = 50,
    n_seg_min: int = 20,
    compactness: float = 10.0,
    bg_aug_enable: bool = True,
    bg_bleed_into_box_ratio: float = 0.07,
    bg_black_dot_prob: float = 0.0005,
    bg_black_dot_size_min: int = 2,
    bg_black_dot_size_max: int = 4,
    bg_blur_sigma_min: float = 1.2,
    bg_blur_sigma_max: float = 2.8,
    bg_brightness: float = 0.45,
    bg_contrast: float = 0.55,
    bg_saturation: float = 0.50,
    bg_hue: float = 0.3,
    target_expand_ratio: float = 0.0,
    fill_blur_sigma: float = 0.0,
    return_details: bool = False,
):
    if yolo_helper is None:
        raise ValueError("occamix_bgfill_data requires a valid YoloCutMixHelper")

    bsz, _, h, w = images.shape
    rng = random

    pre_fill_images = []
    background_masks = []
    foreground_masks = []
    stats = {
        "valid_box_samples": 0,
        "bg_aug_applied_samples": 0,
        "applied_samples": 0,
        "skipped_no_box": 0,
        "skipped_no_background": 0,
        "skipped_empty_target": 0,
        "bg_pool_ratio_sum": 0.0,
        "target_mask_ratio_sum": 0.0,
        "target_intersection_ratio_sum": 0.0,
    }

    for i in range(bsz):
        shifted, shift_info = maybe_center_shift_image(images[i], yolo_helper, img_paths[i], rng)
        foreground_box = shift_info.get("foreground_box")
        if foreground_box is None:
            pre_fill_images.append(shifted)
            background_masks.append(torch.zeros((h, w), dtype=torch.bool, device=images.device))
            foreground_masks.append(torch.zeros((h, w), dtype=torch.bool, device=images.device))
            stats["skipped_no_box"] += 1
            continue

        stats["valid_box_samples"] += 1
        foreground_mask = _box_to_mask(foreground_box, h=h, w=w, device=images.device)
        if bg_aug_enable:
            shifted, bg_info = apply_background_corruption(
                shifted,
                foreground_box,
                rng,
                bleed_ratio=float(max(0.0, bg_bleed_into_box_ratio)),
                black_dot_prob=float(max(0.0, bg_black_dot_prob)),
                black_dot_size_min=int(max(1, bg_black_dot_size_min)),
                black_dot_size_max=int(max(1, bg_black_dot_size_max)),
                blur_sigma_min=float(max(0.0, bg_blur_sigma_min)),
                blur_sigma_max=float(max(0.0, bg_blur_sigma_max)),
                brightness=float(max(0.0, bg_brightness)),
                contrast=float(max(0.0, bg_contrast)),
                saturation=float(max(0.0, bg_saturation)),
                hue=float(max(0.0, min(0.5, bg_hue))),
            )
            background_mask = bg_info.get("background_mask")
            if bg_info.get("bg_aug_applied", False):
                stats["bg_aug_applied_samples"] += 1
        else:
            background_mask = build_background_mask(
                h=h,
                w=w,
                foreground_box=foreground_box,
                bleed_ratio=float(max(0.0, bg_bleed_into_box_ratio)),
                rng=rng,
                device=images.device,
            )

        if background_mask is None:
            background_mask = torch.zeros((h, w), dtype=torch.bool, device=images.device)

        pre_fill_images.append(shifted)
        background_masks.append(background_mask)
        foreground_masks.append(foreground_mask)
        stats["bg_pool_ratio_sum"] += float(background_mask.float().mean().item())

    pre_fill_images_t = torch.stack(pre_fill_images, dim=0)
    background_masks_t = torch.stack(background_masks, dim=0)
    foreground_masks_t = torch.stack(foreground_masks, dim=0)

    occamix_masks, _ = generate_occamix_masks(
        pre_fill_images_t,
        model,
        n_top=n_top,
        n_seg_max=n_seg_max,
        n_seg_min=n_seg_min,
        compactness=compactness,
    )

    target_masks = occamix_masks & foreground_masks_t
    if target_expand_ratio > 0:
        expanded_masks = []
        for i in range(bsz):
            expanded_mask = _expand_mask(target_masks[i], target_expand_ratio)
            expanded_masks.append(expanded_mask & foreground_masks_t[i])
        target_masks = torch.stack(expanded_masks, dim=0)

    filled_images = pre_fill_images_t.clone()
    applied_mask = torch.zeros((bsz,), dtype=torch.bool, device=images.device)
    for i in range(bsz):
        target_mask = target_masks[i]
        background_mask = background_masks_t[i]
        occ_ratio = float(occamix_masks[i].float().mean().item())
        target_ratio = float(target_mask.float().mean().item())
        stats["target_mask_ratio_sum"] += target_ratio
        stats["target_intersection_ratio_sum"] += target_ratio / max(occ_ratio, 1e-8) if occ_ratio > 0 else 0.0

        if not foreground_masks_t[i].any():
            continue
        if not target_mask.any():
            stats["skipped_empty_target"] += 1
            continue
        if not background_mask.any():
            stats["skipped_no_background"] += 1
            continue

        filled, fill_info = fill_mask_from_background(
            pre_fill_images_t[i],
            target_mask,
            background_mask,
            rng=rng,
            fill_blur_sigma=float(max(0.0, fill_blur_sigma)),
        )
        filled_images[i] = filled
        if fill_info["applied"] > 0:
            applied_mask[i] = True
            stats["applied_samples"] += 1

    labels_a = labels
    labels_b = labels
    lam_batch = torch.ones((bsz,), dtype=images.dtype, device=images.device)

    if return_details:
        details = {
            "pre_fill_images": pre_fill_images_t,
            "occamix_masks": occamix_masks,
            "target_masks": target_masks,
            "background_masks": background_masks_t,
            "foreground_masks": foreground_masks_t,
            "applied_mask": applied_mask,
            "stats": stats,
        }
        return filled_images, labels_a, labels_b, lam_batch, details

    return filled_images, labels_a, labels_b, lam_batch


def occamix_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    model,
    n_top: int = 6,
    n_seg_max: int = 50,
    n_seg_min: int = 20,
    compactness: float = 10.0,
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

    Returns:
        mixed_images: [B, C, H, W]
        labels_a:     原始标签
        labels_b:     置乱标签
        lam_batch:    [B] 每个样本的混合面积比例
    """
    bsz, _, h_img, w_img = images.shape
    rand_idx = torch.randperm(bsz, device=images.device)

    labels_a = labels
    labels_b = labels[rand_idx]
    mixed_images = images.clone()
    occamix_masks, lam_batch = generate_occamix_masks(
        images,
        model,
        n_top=n_top,
        n_seg_max=n_seg_max,
        n_seg_min=n_seg_min,
        compactness=compactness,
        source_indices=rand_idx,
    )
    for i in range(bsz):
        if occamix_masks[i].any():
            mixed_images[i, :, occamix_masks[i]] = images[rand_idx[i], :, occamix_masks[i]]
    return mixed_images, labels_a, labels_b, lam_batch
