import math
import os
import random
import hashlib
import json
import csv
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from tqdm import tqdm
from PIL import Image

from config2 import Config2
from augmentation import generate_occamix_masks
from dataset2 import (
    get_train2_dataloaders,
    get_singleview_pseudo_dataloader,
    get_singleview_train2_dataloaders,
    get_singleview_unlabeled_dataloader,
)
from loss import FocalLoss
from metrics import compute_class_metrics, print_class_metrics, print_confusion_matrix
from model import build_train2_model
from utils import (
    AverageMeter,
    accuracy,
    get_current_time,
    load_checkpoint,
    log_training,
    save_checkpoint,
    save_error_samples,
    set_seed,
)
from yolo_cutmix import YoloCutMixHelper


class PairStatsProxy:
    def __init__(self):
        self.pair_random = 0
        self.pair_threshold_matched = 0
        self.pair_threshold_fallback_random = 0
        self.pair_no_area_fallback = 0


def resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    script_dir = Path(__file__).resolve().parent
    candidates = [
        (Path.cwd() / path).resolve(),
        (script_dir / path).resolve(),
    ]
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def normalize_rgb_batch(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def prepare_model_inputs(raw_images: torch.Tensor, detail_images: torch.Tensor, config):
    raw_images = normalize_rgb_batch(raw_images)
    detail_images = normalize_rgb_batch(detail_images)
    model_variant = getattr(config, "MODEL_VARIANT", "dual_branch")
    if model_variant == "single_backbone_6ch":
        return torch.cat([raw_images, detail_images], dim=1), None
    return raw_images, detail_images


def prepare_single_view_input(images: torch.Tensor) -> torch.Tensor:
    return normalize_rgb_batch(images)


def _serialize_config_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serialize_config_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_config_value(v) for k, v in value.items()}
    return str(value)


def snapshot_config(config) -> Dict[str, object]:
    out = {}
    for name in dir(config):
        if not name.isupper():
            continue
        out[name] = _serialize_config_value(getattr(config, name))
    return out


def histogram_summary(values: Sequence[float], bins: Sequence[float]) -> Dict[str, object]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "histogram": [],
        }
    hist, edges = np.histogram(np.asarray(values, dtype=np.float32), bins=np.asarray(bins, dtype=np.float32))
    histogram = []
    for idx, count in enumerate(hist.tolist()):
        histogram.append(
            {
                "left": float(edges[idx]),
                "right": float(edges[idx + 1]),
                "count": int(count),
            }
        )
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "histogram": histogram,
    }


def compute_entropy_batch(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)


def compute_hard_one_hot_loss(logits: torch.Tensor, class_indices: torch.Tensor) -> torch.Tensor:
    targets = F.one_hot(class_indices, num_classes=logits.size(1)).to(dtype=logits.dtype)
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def _compute_mixed_one_hot_loss(logits: torch.Tensor, labels_a, labels_b, lam) -> torch.Tensor:
    num_classes = logits.size(1)
    targets_a = F.one_hot(labels_a, num_classes=num_classes).to(dtype=logits.dtype)
    targets_b = F.one_hot(labels_b, num_classes=num_classes).to(dtype=logits.dtype)
    log_probs = F.log_softmax(logits, dim=1)

    if isinstance(lam, torch.Tensor):
        lam_batch = lam.to(device=logits.device, dtype=logits.dtype)
        if lam_batch.dim() == 0:
            lam_batch = lam_batch.expand(logits.size(0))
        lam_batch = lam_batch.view(-1, 1)
        mixed_targets = lam_batch * targets_a + (1.0 - lam_batch) * targets_b
    else:
        lam_scalar = float(lam)
        mixed_targets = lam_scalar * targets_a + (1.0 - lam_scalar) * targets_b

    return -(mixed_targets * log_probs).sum(dim=1).mean()


def resolve_occamix_cam_model(train_model, cam_model):
    return cam_model if cam_model is not None else train_model


def _load_model_state_from_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get("model_state_dict", checkpoint)
    if any(k.startswith("module.") for k in model_state.keys()):
        model_state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in model_state.items()
        }
    model.load_state_dict(model_state)


def maybe_build_defect_cam_model(config, device: torch.device, checkpoint_attr: str = "DEFECT_CAM_CHECKPOINT_PATH"):
    cam_checkpoint = str(getattr(config, checkpoint_attr, "") or "").strip()
    if not cam_checkpoint:
        return None

    cam_checkpoint_path = resolve_existing_path(cam_checkpoint)
    if not cam_checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_attr} not found: {cam_checkpoint_path}")

    cam_model = build_train2_model(config, pretrained=False)
    cam_model = cam_model.to(device)
    _load_model_state_from_checkpoint(cam_model, str(cam_checkpoint_path), device)
    cam_model.eval()
    for param in cam_model.parameters():
        param.requires_grad = False
    print(f"Using preset OcCaMix CAM model from {checkpoint_attr}: {cam_checkpoint_path}")
    return cam_model


class SavedSingleViewValDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, int, str]], image_size: int):
        self.samples = list(samples)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        saved_path, label, original_path = self.samples[idx]
        image = Image.open(saved_path).convert("RGB")
        image = self.transform(image)
        return image, label, original_path


def build_helper_from_config(config, train_dir: str) -> YoloCutMixHelper:
    return YoloCutMixHelper(
        cache_path=config.YOLO_CACHE_PATH,
        train_dir=train_dir,
        key_mode=config.YOLO_KEY_MODE,
        fallback_mode="skip",
        min_box_area_ratio=config.YOLO_MIN_BOX_AREA_RATIO,
        max_box_area_ratio=config.YOLO_MAX_BOX_AREA_RATIO,
        sector_center_jitter_ratio=getattr(config, "SECTOR_CENTER_JITTER_RATIO", 0.0),
        enable_recenter_shift=getattr(config, "ENABLE_CENTER_SHIFT", True),
        center_tolerance_ratio=getattr(config, "CENTER_TOLERANCE_RATIO", 0.0),
        debug_log=False,
        pair_use_area_match=getattr(config, "PAIR_USE_AREA_MATCH", True),
        pair_random_prob=getattr(config, "PAIR_RANDOM_PROB", 0.1),
        pair_area_ratio_min=getattr(config, "PAIR_AREA_RATIO_MIN", 0.4),
        pair_area_ratio_max=getattr(config, "PAIR_AREA_RATIO_MAX", 2.5),
        pair_area_ratio_min_target=getattr(config, "PAIR_AREA_RATIO_MIN", 0.4),
        pair_area_ratio_max_target=getattr(config, "PAIR_AREA_RATIO_MAX", 2.5),
        pair_schedule_start_ratio=0.0,
        pair_schedule_end_ratio=0.0,
    )


def build_helper_with_cache(config, train_dir: str, cache_path: str) -> YoloCutMixHelper:
    return YoloCutMixHelper(
        cache_path=cache_path,
        train_dir=train_dir,
        key_mode=config.YOLO_KEY_MODE,
        fallback_mode="skip",
        min_box_area_ratio=config.YOLO_MIN_BOX_AREA_RATIO,
        max_box_area_ratio=config.YOLO_MAX_BOX_AREA_RATIO,
        sector_center_jitter_ratio=getattr(config, "SECTOR_CENTER_JITTER_RATIO", 0.0),
        enable_recenter_shift=getattr(config, "ENABLE_CENTER_SHIFT", True),
        center_tolerance_ratio=getattr(config, "CENTER_TOLERANCE_RATIO", 0.0),
        debug_log=False,
        pair_use_area_match=getattr(config, "PAIR_USE_AREA_MATCH", True),
        pair_random_prob=getattr(config, "PAIR_RANDOM_PROB", 0.1),
        pair_area_ratio_min=getattr(config, "PAIR_AREA_RATIO_MIN", 0.4),
        pair_area_ratio_max=getattr(config, "PAIR_AREA_RATIO_MAX", 2.5),
        pair_area_ratio_min_target=getattr(config, "PAIR_AREA_RATIO_MIN", 0.4),
        pair_area_ratio_max_target=getattr(config, "PAIR_AREA_RATIO_MAX", 2.5),
        pair_schedule_start_ratio=0.0,
        pair_schedule_end_ratio=0.0,
    )


def lookup_valid_boxes(helper: YoloCutMixHelper, img_path: str, h: int, w: int) -> List[Tuple[int, int, int, int]]:
    entry = helper._lookup_entry(img_path)
    boxes_raw, orig_size = helper._parse_entry(entry)
    return helper._build_valid_boxes(boxes_raw, orig_size, h=h, w=w)


def maybe_center_shift_image(
    image: torch.Tensor,
    helper: YoloCutMixHelper,
    img_path: str,
    rng: random.Random,
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
    rng: random.Random,
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
    rng: random.Random,
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
    rng: random.Random,
    config,
) -> torch.Tensor:
    if foreground_box is None:
        return image

    _, h, w = image.shape
    background_mask = build_background_mask(
        h,
        w,
        foreground_box,
        float(max(0.0, getattr(config, "BG_BLEED_INTO_BOX_RATIO", 0.0))),
        rng,
        image.device,
    )
    if float(background_mask.float().mean().item()) <= 0.0:
        return image

    corrupted = image.clone()
    blur_sigma_min = float(max(0.0, getattr(config, "BG_BLUR_SIGMA_MIN", 0.0)))
    blur_sigma_max = float(max(blur_sigma_min, getattr(config, "BG_BLUR_SIGMA_MAX", blur_sigma_min)))
    if blur_sigma_max > 0:
        sigma = rng.uniform(blur_sigma_min, blur_sigma_max)
        kernel_size = max(3, int(round(sigma * 4)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        corrupted = TF.gaussian_blur(corrupted, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    brightness = float(max(0.0, getattr(config, "BG_BRIGHTNESS", 0.0)))
    contrast = float(max(0.0, getattr(config, "BG_CONTRAST", 0.0)))
    saturation = float(max(0.0, getattr(config, "BG_SATURATION", 0.0)))
    hue = float(max(0.0, min(0.5, getattr(config, "BG_HUE", 0.0))))
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
        dot_prob=float(max(0.0, getattr(config, "BG_BLACK_DOT_PROB", 0.0))),
        dot_size_min=int(max(1, getattr(config, "BG_BLACK_DOT_SIZE_MIN", 1))),
        dot_size_max=int(max(1, getattr(config, "BG_BLACK_DOT_SIZE_MAX", 1))),
    )

    return torch.where(background_mask.unsqueeze(0).expand_as(image), corrupted, image)


def apply_synchronized_rotation(
    raw_images: torch.Tensor,
    detail_images: torch.Tensor,
    rng: random.Random,
    max_degrees: float,
    fill: float,
):
    if max_degrees <= 0:
        return raw_images, detail_images

    rotated_raw = []
    rotated_detail = []
    for raw_img, detail_img in zip(raw_images, detail_images):
        angle = rng.uniform(-max_degrees, max_degrees)
        rotated_raw.append(
            TF.rotate(
                raw_img,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=float(fill),
            )
        )
        rotated_detail.append(
            TF.rotate(
                detail_img,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=float(fill),
            )
        )
    return torch.stack(rotated_raw, dim=0), torch.stack(rotated_detail, dim=0)


def apply_synchronized_flip(
    raw_images: torch.Tensor,
    detail_images: torch.Tensor,
    vertical: bool = False,
):
    flipped_raw = []
    flipped_detail = []
    for raw_img, detail_img in zip(raw_images, detail_images):
        if vertical:
            flipped_raw.append(TF.vflip(raw_img))
            flipped_detail.append(TF.vflip(detail_img))
        else:
            flipped_raw.append(TF.hflip(raw_img))
            flipped_detail.append(TF.hflip(detail_img))
    return torch.stack(flipped_raw, dim=0), torch.stack(flipped_detail, dim=0)


def apply_rotation_batch(
    images: torch.Tensor,
    rng: random.Random,
    max_degrees: float,
    fill: float,
):
    if max_degrees <= 0:
        return images

    rotated = []
    for image in images:
        angle = rng.uniform(-max_degrees, max_degrees)
        rotated.append(
            TF.rotate(
                image,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=float(fill),
            )
        )
    return torch.stack(rotated, dim=0)


def apply_flip_batch(images: torch.Tensor, vertical: bool = False):
    flipped = []
    for image in images:
        flipped.append(TF.vflip(image) if vertical else TF.hflip(image))
    return torch.stack(flipped, dim=0)


def expand_binary_mask(mask: torch.Tensor, expand_ratio: float) -> torch.Tensor:
    if expand_ratio <= 0:
        return mask
    h, w = mask.shape
    radius = int(round(float(min(h, w)) * expand_ratio))
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    expanded = nn.functional.max_pool2d(
        mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )
    return expanded.squeeze(0).squeeze(0) > 0


def extract_border_pixels(image: torch.Tensor, border_width: int) -> torch.Tensor:
    _, h, w = image.shape
    border_width = max(1, int(border_width))
    border_width = min(border_width, max(1, min(h, w) // 2))
    border_mask = torch.zeros((h, w), dtype=torch.bool, device=image.device)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    return image[:, border_mask].transpose(0, 1).contiguous()


def compute_quantized_border_mode_color(image: torch.Tensor, border_width: int, quantize: int) -> torch.Tensor:
    border_pixels = extract_border_pixels(image, border_width=border_width)
    if border_pixels.numel() == 0:
        return image.mean(dim=(1, 2))

    quantize = max(1, int(quantize))
    pixels_255 = torch.clamp(torch.round(border_pixels * 255.0), 0.0, 255.0).to(dtype=torch.int32)
    quantized = torch.clamp((pixels_255 // quantize) * quantize, 0, 255)
    quantized_np = quantized.detach().cpu().numpy()
    unique_colors, counts = np.unique(quantized_np, axis=0, return_counts=True)
    non_zero_mask = np.any(unique_colors != 0, axis=1)
    if np.any(non_zero_mask):
        unique_colors = unique_colors[non_zero_mask]
        counts = counts[non_zero_mask]
    mode_color = unique_colors[int(np.argmax(counts))]
    return torch.tensor(mode_color, dtype=image.dtype, device=image.device) / 255.0


def fill_mask_with_mode_color(
    image: torch.Tensor,
    target_mask: torch.Tensor,
    fill_color: torch.Tensor,
    fill_blur_sigma: float = 0.0,
) -> torch.Tensor:
    if not target_mask.any():
        return image

    output = image.clone()
    fill = fill_color.view(3, 1, 1).expand_as(output)
    output = torch.where(target_mask.unsqueeze(0), fill, output)

    if fill_blur_sigma > 0:
        sigma = float(fill_blur_sigma)
        kernel_size = max(3, int(round(sigma * 4)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = TF.gaussian_blur(output, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        output = torch.where(target_mask.unsqueeze(0), blurred, output)
    return output


def _stable_path_seed(base_seed: int, path: str) -> int:
    digest = hashlib.md5(path.encode("utf-8")).hexdigest()[:8]
    return int(base_seed) + int(digest, 16)


def apply_singleview_defect_batch(
    images: torch.Tensor,
    image_paths: Sequence[str],
    helper: YoloCutMixHelper,
    model,
    config,
    seed_offset: int = 0,
    cam_model=None,
) -> torch.Tensor:
    bsz = images.shape[0]
    base_seed = int(getattr(config, "SEED", 17)) + int(seed_offset)
    base_images = []
    for i in range(bsz):
        rng = random.Random(_stable_path_seed(base_seed, str(image_paths[i])))
        shifted, _ = maybe_center_shift_image(images[i], helper, image_paths[i], rng)
        base_images.append(shifted)

    base_t = torch.stack(base_images, dim=0)
    saved_random_state = random.getstate()
    batch_seed_material = "|".join(str(p) for p in image_paths)
    random.seed(_stable_path_seed(base_seed, batch_seed_material))
    try:
        defect_masks, _ = generate_occamix_masks(
            base_t,
            resolve_occamix_cam_model(model, cam_model),
            n_top=int(getattr(config, "DEFECT_VAL_N_TOP", getattr(config, "DEFECT_N_TOP", 3))),
            n_seg_max=int(getattr(config, "DEFECT_VAL_SEG_MAX", getattr(config, "DEFECT_SEG_MAX", 80))),
            n_seg_min=int(getattr(config, "DEFECT_VAL_SEG_MIN", getattr(config, "DEFECT_SEG_MIN", 40))),
            compactness=float(getattr(config, "DEFECT_VAL_COMPACTNESS", getattr(config, "DEFECT_COMPACTNESS", 5.0))),
        )
    finally:
        random.setstate(saved_random_state)

    out_t = base_t.clone()
    for i in range(bsz):
        target_mask = defect_masks[i]
        target_mask = expand_binary_mask(
            target_mask,
            float(max(0.0, getattr(config, "DEFECT_VAL_TARGET_EXPAND_RATIO", getattr(config, "DEFECT_TARGET_EXPAND_RATIO", 0.0)))),
        )
        if not target_mask.any():
            continue
        fill_color = compute_quantized_border_mode_color(
            base_t[i],
            border_width=int(max(1, getattr(config, "DEFECT_VAL_BORDER_WIDTH", getattr(config, "DEFECT_BORDER_WIDTH", 4)))),
            quantize=int(max(1, getattr(config, "DEFECT_VAL_BORDER_QUANTIZE", getattr(config, "DEFECT_BORDER_QUANTIZE", 16)))),
        )
        out_t[i] = fill_mask_with_mode_color(
            base_t[i],
            target_mask,
            fill_color,
            fill_blur_sigma=float(max(0.0, getattr(config, "DEFECT_VAL_FILL_BLUR_SIGMA", getattr(config, "DEFECT_FILL_BLUR_SIGMA", 0.0)))),
        )
    return out_t


def materialize_singleview_defect_val_loader(
    clean_val_loader,
    model,
    helper,
    config,
    device,
    cam_model=None,
):
    output_dir = Path(config.CHECKPOINT_DIR) / "defect_val_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    saved_samples: List[Tuple[str, int, str]] = []
    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch_idx, (images, labels, image_paths) in enumerate(tqdm(clean_val_loader, desc="[Build Defect Val]")):
            images = images.to(device, non_blocking=True)
            defect_images = apply_singleview_defect_batch(
                images,
                image_paths,
                helper=helper,
                model=model,
                config=config,
                seed_offset=99991,
                cam_model=cam_model,
            ).cpu()

            for sample_idx, (image, label, original_path) in enumerate(zip(defect_images, labels, image_paths)):
                suffix = Path(original_path).suffix.lower() or ".png"
                stem = Path(original_path).stem
                save_name = f"{batch_idx:04d}_{sample_idx:02d}_{stem}{suffix}"
                save_path = output_dir / save_name
                TF.to_pil_image(image).save(save_path)
                saved_samples.append((str(save_path), int(label), str(original_path)))

    if model_was_training:
        model.train()

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("saved_path,label,original_path\n")
        for saved_path, label, original_path in saved_samples:
            f.write(f"{saved_path},{label},{original_path}\n")

    loader_kwargs = {
        "num_workers": 0,
        "pin_memory": bool(getattr(config, "PIN_MEMORY", True)),
    }
    dataset = SavedSingleViewValDataset(saved_samples, image_size=int(getattr(config, "IMAGE_SIZE", 224)))
    loader = DataLoader(
        dataset,
        batch_size=int(getattr(config, "BATCH_SIZE", 32)),
        shuffle=False,
        **loader_kwargs,
    )
    return loader, str(output_dir)


def _compute_mixed_loss(logits, labels_a, labels_b, lam, criterion, config):
    if isinstance(lam, torch.Tensor) and config.LOSS_TYPE == "cross_entropy":
        lam_batch = lam.to(device=logits.device, dtype=logits.dtype).view(-1)
        loss_a = nn.functional.cross_entropy(
            logits,
            labels_a,
            label_smoothing=config.LABEL_SMOOTHING,
            reduction="none",
        )
        loss_b = nn.functional.cross_entropy(
            logits,
            labels_b,
            label_smoothing=config.LABEL_SMOOTHING,
            reduction="none",
        )
        return (lam_batch * loss_a + (1.0 - lam_batch) * loss_b).mean()

    if isinstance(lam, torch.Tensor):
        lam_batch = lam.to(device=logits.device, dtype=logits.dtype).view(-1)
        if hasattr(criterion, "reduction"):
            old_reduction = criterion.reduction
            try:
                criterion.reduction = "none"
                loss_a = criterion(logits, labels_a)
                loss_b = criterion(logits, labels_b)
            finally:
                criterion.reduction = old_reduction
            return (lam_batch * loss_a + (1.0 - lam_batch) * loss_b).mean()
        lam_scalar = float(lam_batch.mean().item())
        return lam_scalar * criterion(logits, labels_a) + (1.0 - lam_scalar) * criterion(logits, labels_b)

    return float(lam) * criterion(logits, labels_a) + (1.0 - float(lam)) * criterion(logits, labels_b)


def apply_dualview_augmentation(
    raw_images: torch.Tensor,
    detail_images: torch.Tensor,
    labels: torch.Tensor,
    raw_paths: Sequence[str],
    detail_paths: Sequence[str],
    helper_raw: YoloCutMixHelper,
    helper_detail: YoloCutMixHelper,
    alpha: float,
    apply_background_aug: bool,
    apply_cutmix: bool,
    apply_rotate: bool,
    apply_flip: bool,
    config,
    epoch: int,
    batch_idx: int,
):
    bsz, _, h, w = raw_images.shape
    rng = random.Random(int(getattr(config, "SEED", 17)) + epoch * 100003 + batch_idx)

    raw_base = []
    detail_base = []
    for i in range(bsz):
        raw_shifted, raw_info = maybe_center_shift_image(raw_images[i], helper_raw, raw_paths[i], rng)
        detail_shifted, detail_info = maybe_center_shift_image(detail_images[i], helper_detail, detail_paths[i], rng)
        if apply_background_aug:
            raw_shifted = apply_background_corruption(raw_shifted, raw_info.get("foreground_box"), rng, config)
            detail_shifted = apply_background_corruption(detail_shifted, detail_info.get("foreground_box"), rng, config)
        raw_base.append(raw_shifted)
        detail_base.append(detail_shifted)

    raw_base_t = torch.stack(raw_base, dim=0)
    detail_base_t = torch.stack(detail_base, dim=0)

    raw_out_t = raw_base_t
    detail_out_t = detail_base_t
    labels_a = labels
    labels_b = labels
    lam = 1.0
    stats = {
        "pair_random": 0,
        "pair_threshold_matched": 0,
        "pair_threshold_fallback_random": 0,
        "pair_no_area_fallback": 0,
    }

    if apply_cutmix:
        raw_area_ratios = [
            helper_raw._estimate_area_ratio_from_valid_boxes(
                lookup_valid_boxes(helper_raw, raw_paths[i], h=h, w=w),
                h,
                w,
            )
            for i in range(bsz)
        ]
        pair_stats = PairStatsProxy()
        rand_idx = helper_raw._build_pair_indices(raw_area_ratios, device=raw_images.device, stats=pair_stats)

        lam_scalar = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
        span_ratio = float(np.clip(1.0 - lam_scalar, 0.0, 1.0))
        span_angle = span_ratio * 2.0 * np.pi

        raw_out_t = raw_base_t.clone()
        detail_out_t = detail_base_t.clone()
        lam_batch = []
        for i in range(bsz):
            center_x, center_y = helper_raw._sample_sector_center(h=h, w=w)
            theta_start = float(np.random.uniform(-np.pi, np.pi))
            theta_end = theta_start + span_angle
            raw_mask = helper_raw._build_sector_mask(
                h=h,
                w=w,
                center_x=center_x,
                center_y=center_y,
                theta_start=theta_start,
                theta_end=theta_end,
                device=raw_images.device,
            )
            detail_mask = helper_detail._build_sector_mask(
                h=h,
                w=w,
                center_x=center_x,
                center_y=center_y,
                theta_start=theta_start,
                theta_end=theta_end,
                device=detail_images.device,
            )
            donor_idx = int(rand_idx[i].item())
            raw_out_t[i] = torch.where(raw_mask.unsqueeze(0).expand_as(raw_out_t[i]), raw_base_t[donor_idx], raw_base_t[i])
            detail_out_t[i] = torch.where(detail_mask.unsqueeze(0).expand_as(detail_out_t[i]), detail_base_t[donor_idx], detail_base_t[i])
            lam_batch.append(1.0 - float(raw_mask.float().mean().item()))

        stats = {
            "pair_random": int(pair_stats.pair_random),
            "pair_threshold_matched": int(pair_stats.pair_threshold_matched),
            "pair_threshold_fallback_random": int(pair_stats.pair_threshold_fallback_random),
            "pair_no_area_fallback": int(pair_stats.pair_no_area_fallback),
        }
        labels_b = labels[rand_idx]
        lam = torch.tensor(lam_batch, dtype=raw_images.dtype, device=raw_images.device)

    if apply_rotate:
        raw_out_t, detail_out_t = apply_synchronized_rotation(
            raw_out_t,
            detail_out_t,
            rng=rng,
            max_degrees=float(max(0.0, getattr(config, "ROTATE_DEGREES", 0.0))),
            fill=float(getattr(config, "ROTATE_FILL", 0.0)),
        )

    if apply_flip:
        raw_out_t, detail_out_t = apply_synchronized_flip(
            raw_out_t,
            detail_out_t,
            vertical=bool(getattr(config, "FLIP_VERTICAL", False)),
        )

    return raw_out_t, detail_out_t, labels_a, labels_b, lam, stats


def apply_singleview_augmentation(
    images: torch.Tensor,
    labels: torch.Tensor,
    image_paths: Sequence[str],
    helper: YoloCutMixHelper,
    model,
    alpha: float,
    apply_background_aug: bool,
    apply_defect: bool,
    apply_cutmix: bool,
    apply_rotate: bool,
    apply_flip: bool,
    config,
    epoch: int,
    batch_idx: int,
    cam_model=None,
):
    bsz, _, h, w = images.shape
    rng = random.Random(int(getattr(config, "SEED", 17)) + epoch * 100003 + batch_idx)

    base_images = []
    for i in range(bsz):
        shifted, info = maybe_center_shift_image(images[i], helper, image_paths[i], rng)
        if apply_background_aug:
            shifted = apply_background_corruption(shifted, info.get("foreground_box"), rng, config)
        base_images.append(shifted)

    base_t = torch.stack(base_images, dim=0)
    out_t = base_t
    labels_a = labels
    labels_b = labels
    lam = 1.0
    stats = {
        "pair_random": 0,
        "pair_threshold_matched": 0,
        "pair_threshold_fallback_random": 0,
        "pair_no_area_fallback": 0,
        "defect_applied_samples": 0,
        "defect_skipped_empty_mask": 0,
    }

    if apply_defect:
        defect_masks, _ = generate_occamix_masks(
            base_t,
            resolve_occamix_cam_model(model, cam_model),
            n_top=int(getattr(config, "DEFECT_N_TOP", 3)),
            n_seg_max=int(getattr(config, "DEFECT_SEG_MAX", 80)),
            n_seg_min=int(getattr(config, "DEFECT_SEG_MIN", 40)),
            compactness=float(getattr(config, "DEFECT_COMPACTNESS", 5.0)),
        )
        out_t = base_t.clone()
        for i in range(bsz):
            target_mask = defect_masks[i]
            target_mask = expand_binary_mask(
                target_mask,
                float(max(0.0, getattr(config, "DEFECT_TARGET_EXPAND_RATIO", 0.0))),
            )
            if not target_mask.any():
                stats["defect_skipped_empty_mask"] += 1
                continue
            fill_color = compute_quantized_border_mode_color(
                base_t[i],
                border_width=int(max(1, getattr(config, "DEFECT_BORDER_WIDTH", 4))),
                quantize=int(max(1, getattr(config, "DEFECT_BORDER_QUANTIZE", 16))),
            )
            out_t[i] = fill_mask_with_mode_color(
                base_t[i],
                target_mask,
                fill_color,
                fill_blur_sigma=float(max(0.0, getattr(config, "DEFECT_FILL_BLUR_SIGMA", 0.0))),
            )
            stats["defect_applied_samples"] += 1

    elif apply_cutmix:
        area_ratios = [
            helper._estimate_area_ratio_from_valid_boxes(
                lookup_valid_boxes(helper, image_paths[i], h=h, w=w),
                h,
                w,
            )
            for i in range(bsz)
        ]
        pair_stats = PairStatsProxy()
        rand_idx = helper._build_pair_indices(area_ratios, device=images.device, stats=pair_stats)

        lam_scalar = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
        span_ratio = float(np.clip(1.0 - lam_scalar, 0.0, 1.0))
        span_angle = span_ratio * 2.0 * np.pi

        out_t = base_t.clone()
        lam_batch = []
        for i in range(bsz):
            center_x, center_y = helper._sample_sector_center(h=h, w=w)
            theta_start = float(np.random.uniform(-np.pi, np.pi))
            theta_end = theta_start + span_angle
            mask = helper._build_sector_mask(
                h=h,
                w=w,
                center_x=center_x,
                center_y=center_y,
                theta_start=theta_start,
                theta_end=theta_end,
                device=images.device,
            )
            donor_idx = int(rand_idx[i].item())
            out_t[i] = torch.where(mask.unsqueeze(0).expand_as(out_t[i]), base_t[donor_idx], base_t[i])
            lam_batch.append(1.0 - float(mask.float().mean().item()))

        stats = {
            "pair_random": int(pair_stats.pair_random),
            "pair_threshold_matched": int(pair_stats.pair_threshold_matched),
            "pair_threshold_fallback_random": int(pair_stats.pair_threshold_fallback_random),
            "pair_no_area_fallback": int(pair_stats.pair_no_area_fallback),
        }
        labels_b = labels[rand_idx]
        lam = torch.tensor(lam_batch, dtype=images.dtype, device=images.device)

    if apply_rotate:
        out_t = apply_rotation_batch(
            out_t,
            rng=rng,
            max_degrees=float(max(0.0, getattr(config, "ROTATE_DEGREES", 0.0))),
            fill=float(getattr(config, "ROTATE_FILL", 0.0)),
        )

    if apply_flip:
        out_t = apply_flip_batch(
            out_t,
            vertical=bool(getattr(config, "FLIP_VERTICAL", False)),
        )

    return out_t, labels_a, labels_b, lam, stats


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config, helper_raw, helper_detail):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    aug_stats = {
        "bg_aug_batches": 0,
        "cutmix_batches": 0,
        "rotate_batches": 0,
        "flip_batches": 0,
        "both_aug_batches": 0,
        "bg_only_batches": 0,
        "cutmix_only_batches": 0,
        "rotate_only_batches": 0,
        "flip_only_batches": 0,
        "bg_cutmix_rotate_batches": 0,
        "all_aug_batches": 0,
        "pair_random": 0,
        "pair_threshold_matched": 0,
        "pair_threshold_fallback_random": 0,
        "pair_no_area_fallback": 0,
    }

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train2]")
    for batch_idx, (raw_images, detail_images, labels, raw_paths, detail_paths) in enumerate(pbar):
        raw_images = raw_images.to(device, non_blocking=True)
        detail_images = detail_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        labels_a, labels_b, lam = labels, labels, 1.0
        apply_bg_aug = bool(getattr(config, "BG_AUG_ENABLE", False)) and (
            np.random.rand() < float(getattr(config, "BG_AUG_PROB", 0.0))
        )
        apply_cutmix = bool(getattr(config, "CUTMIX_ENABLE", True)) and (
            np.random.rand() < float(getattr(config, "CUTMIX_PROB", 0.0))
        )
        apply_rotate = bool(getattr(config, "ROTATE_ENABLE", False)) and (
            np.random.rand() < float(getattr(config, "ROTATE_PROB", 0.0))
        )
        apply_flip = bool(getattr(config, "FLIP_ENABLE", False)) and (
            np.random.rand() < float(getattr(config, "FLIP_PROB", 0.0))
        )
        if apply_bg_aug:
            aug_stats["bg_aug_batches"] += 1
        if apply_cutmix:
            aug_stats["cutmix_batches"] += 1
        if apply_rotate:
            aug_stats["rotate_batches"] += 1
        if apply_flip:
            aug_stats["flip_batches"] += 1
        if apply_bg_aug and apply_cutmix:
            aug_stats["both_aug_batches"] += 1
        if apply_bg_aug and not apply_cutmix and not apply_rotate and not apply_flip:
            aug_stats["bg_only_batches"] += 1
        if apply_cutmix and not apply_bg_aug and not apply_rotate and not apply_flip:
            aug_stats["cutmix_only_batches"] += 1
        if apply_rotate and not apply_bg_aug and not apply_cutmix and not apply_flip:
            aug_stats["rotate_only_batches"] += 1
        if apply_flip and not apply_bg_aug and not apply_cutmix and not apply_rotate:
            aug_stats["flip_only_batches"] += 1
        if apply_bg_aug and apply_cutmix and apply_rotate:
            aug_stats["bg_cutmix_rotate_batches"] += 1
        if apply_bg_aug and apply_cutmix and apply_rotate and apply_flip:
            aug_stats["all_aug_batches"] += 1

        if apply_bg_aug or apply_cutmix or apply_rotate or apply_flip:
            raw_images, detail_images, labels_a, labels_b, lam, batch_stats = apply_dualview_augmentation(
                raw_images,
                detail_images,
                labels,
                list(raw_paths),
                list(detail_paths),
                helper_raw,
                helper_detail,
                alpha=float(getattr(config, "CUTMIX_ALPHA", 1.0)),
                apply_background_aug=apply_bg_aug,
                apply_cutmix=apply_cutmix,
                apply_rotate=apply_rotate,
                apply_flip=apply_flip,
                config=config,
                epoch=epoch,
                batch_idx=batch_idx,
            )
            for key in ("pair_random", "pair_threshold_matched", "pair_threshold_fallback_random", "pair_no_area_fallback"):
                aug_stats[key] += int(batch_stats.get(key, 0))

        model_input, detail_model_input = prepare_model_inputs(raw_images, detail_images, config)

        optimizer.zero_grad()
        if detail_model_input is None:
            outputs = model(model_input)
        else:
            outputs = model(model_input, detail_model_input)
        fusion_loss = _compute_mixed_loss(outputs["fusion_logits"], labels_a, labels_b, lam, criterion, config)
        loss = fusion_loss
        if "raw_logits" in outputs and "detail_logits" in outputs:
            raw_aux_loss = _compute_mixed_loss(outputs["raw_logits"], labels_a, labels_b, lam, criterion, config)
            detail_aux_loss = _compute_mixed_loss(outputs["detail_logits"], labels_a, labels_b, lam, criterion, config)
            loss = (
                fusion_loss
                + float(getattr(config, "RAW_AUX_LOSS_WEIGHT", 0.3)) * raw_aux_loss
                + float(getattr(config, "DETAIL_AUX_LOSS_WEIGHT", 0.3)) * detail_aux_loss
            )
        loss.backward()
        optimizer.step()

        acc1 = accuracy(outputs["fusion_logits"], labels, topk=(1,))[0]
        losses.update(loss.item(), model_input.size(0))
        top1.update(acc1.item(), model_input.size(0))
        pbar.set_postfix({"Loss": f"{losses.avg:.4f}", "Acc": f"{top1.avg:.2f}%"})

    return losses.avg, top1.avg, aug_stats


@torch.no_grad()
def validate(model, val_loader, criterion, device, config):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_preds = []
    all_labels = []
    error_samples = []

    pbar = tqdm(val_loader, desc="[Validate2]")
    for raw_images, detail_images, labels, raw_paths, _ in pbar:
        raw_images = raw_images.to(device, non_blocking=True)
        detail_images = detail_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model_input, detail_model_input = prepare_model_inputs(raw_images, detail_images, config)
        if detail_model_input is None:
            outputs = model(model_input)
        else:
            outputs = model(model_input, detail_model_input)
        fusion_logits = outputs["fusion_logits"]
        loss = criterion(fusion_logits, labels)
        acc1 = accuracy(fusion_logits, labels, topk=(1,))[0]

        losses.update(loss.item(), model_input.size(0))
        top1.update(acc1.item(), model_input.size(0))

        preds = fusion_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        probs = torch.softmax(fusion_logits, dim=1)
        max_probs = probs.max(dim=1)
        for pred, label, img_path, max_prob in zip(
            preds.cpu().numpy(),
            labels.cpu().numpy(),
            raw_paths,
            max_probs.values.cpu().numpy(),
        ):
            if pred != label:
                error_samples.append(
                    {
                        "img_path": img_path,
                        "pred_label": int(pred),
                        "true_label": int(label),
                        "confidence": float(max_prob),
                    }
                )

        pbar.set_postfix({"Loss": f"{losses.avg:.4f}", "Acc": f"{top1.avg:.2f}%"})

    return losses.avg, top1.avg, all_preds, all_labels, error_samples


def train_one_epoch_singleview(model, train_loader, criterion, optimizer, device, epoch, config, helper, cam_model=None):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    aug_stats = {
        "bg_aug_batches": 0,
        "cutmix_batches": 0,
        "defect_batches": 0,
        "rotate_batches": 0,
        "flip_batches": 0,
        "both_aug_batches": 0,
        "bg_only_batches": 0,
        "cutmix_only_batches": 0,
        "defect_only_batches": 0,
        "rotate_only_batches": 0,
        "flip_only_batches": 0,
        "bg_cutmix_rotate_batches": 0,
        "all_aug_batches": 0,
        "cutmix_defect_conflict_batches": 0,
        "defect_applied_samples": 0,
        "defect_skipped_empty_mask": 0,
        "pair_random": 0,
        "pair_threshold_matched": 0,
        "pair_threshold_fallback_random": 0,
        "pair_no_area_fallback": 0,
    }

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train2-Single]")
    for batch_idx, (images, labels, image_paths) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        labels_a, labels_b, lam = labels, labels, 1.0
        apply_bg_aug = bool(getattr(config, "BG_AUG_ENABLE", False)) and (
            np.random.rand() < float(getattr(config, "BG_AUG_PROB", 0.0))
        )
        apply_cutmix = bool(getattr(config, "CUTMIX_ENABLE", True)) and (
            np.random.rand() < float(getattr(config, "CUTMIX_PROB", 0.0))
        )
        apply_defect = False
        if not apply_cutmix:
            apply_defect = bool(getattr(config, "DEFECT_ENABLE", False)) and (
                np.random.rand() < float(getattr(config, "DEFECT_PROB", 0.0))
            )
        apply_rotate = bool(getattr(config, "ROTATE_ENABLE", False)) and (
            np.random.rand() < float(getattr(config, "ROTATE_PROB", 0.0))
        )
        apply_flip = bool(getattr(config, "FLIP_ENABLE", False)) and (
            np.random.rand() < float(getattr(config, "FLIP_PROB", 0.0))
        )
        if apply_bg_aug:
            aug_stats["bg_aug_batches"] += 1
        if apply_cutmix:
            aug_stats["cutmix_batches"] += 1
        if apply_defect:
            aug_stats["defect_batches"] += 1
        if apply_rotate:
            aug_stats["rotate_batches"] += 1
        if apply_flip:
            aug_stats["flip_batches"] += 1
        if apply_bg_aug and apply_cutmix:
            aug_stats["both_aug_batches"] += 1
        if apply_bg_aug and not apply_cutmix and not apply_defect and not apply_rotate and not apply_flip:
            aug_stats["bg_only_batches"] += 1
        if apply_cutmix and not apply_bg_aug and not apply_defect and not apply_rotate and not apply_flip:
            aug_stats["cutmix_only_batches"] += 1
        if apply_defect and not apply_bg_aug and not apply_cutmix and not apply_rotate and not apply_flip:
            aug_stats["defect_only_batches"] += 1
        if apply_rotate and not apply_bg_aug and not apply_cutmix and not apply_defect and not apply_flip:
            aug_stats["rotate_only_batches"] += 1
        if apply_flip and not apply_bg_aug and not apply_cutmix and not apply_defect and not apply_rotate:
            aug_stats["flip_only_batches"] += 1
        if apply_bg_aug and apply_cutmix and apply_rotate:
            aug_stats["bg_cutmix_rotate_batches"] += 1
        if apply_bg_aug and (apply_cutmix or apply_defect) and apply_rotate and apply_flip:
            aug_stats["all_aug_batches"] += 1

        if apply_bg_aug or apply_cutmix or apply_defect or apply_rotate or apply_flip:
            images, labels_a, labels_b, lam, batch_stats = apply_singleview_augmentation(
                images,
                labels,
                list(image_paths),
                helper,
                model,
                alpha=float(getattr(config, "CUTMIX_ALPHA", 1.0)),
                apply_background_aug=apply_bg_aug,
                apply_defect=apply_defect,
                apply_cutmix=apply_cutmix,
                apply_rotate=apply_rotate,
                apply_flip=apply_flip,
                config=config,
                epoch=epoch,
                batch_idx=batch_idx,
                cam_model=cam_model,
            )
            for key in ("pair_random", "pair_threshold_matched", "pair_threshold_fallback_random", "pair_no_area_fallback"):
                aug_stats[key] += int(batch_stats.get(key, 0))
            aug_stats["defect_applied_samples"] += int(batch_stats.get("defect_applied_samples", 0))
            aug_stats["defect_skipped_empty_mask"] += int(batch_stats.get("defect_skipped_empty_mask", 0))

        model_input = prepare_single_view_input(images)

        optimizer.zero_grad()
        outputs = model(model_input)
        loss = _compute_mixed_loss(outputs["fusion_logits"], labels_a, labels_b, lam, criterion, config)
        loss.backward()
        optimizer.step()

        acc1 = accuracy(outputs["fusion_logits"], labels, topk=(1,))[0]
        losses.update(loss.item(), model_input.size(0))
        top1.update(acc1.item(), model_input.size(0))
        pbar.set_postfix({"Loss": f"{losses.avg:.4f}", "Acc": f"{top1.avg:.2f}%"})

    return losses.avg, top1.avg, aug_stats


@torch.no_grad()
def validate_singleview(model, val_loader, criterion, device, helper=None, config=None, apply_defect_eval=False, cam_model=None):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_preds = []
    all_labels = []
    error_samples = []

    pbar = tqdm(val_loader, desc="[Validate2-Single]")
    for images, labels, image_paths in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if apply_defect_eval:
            if helper is None or config is None:
                raise ValueError("Defect validation requires helper and config")
            images = apply_singleview_defect_batch(
                images,
                image_paths,
                helper=helper,
                model=model,
                config=config,
                cam_model=cam_model,
            )

        model_input = prepare_single_view_input(images)
        outputs = model(model_input)
        fusion_logits = outputs["fusion_logits"]
        loss = criterion(fusion_logits, labels)
        acc1 = accuracy(fusion_logits, labels, topk=(1,))[0]

        losses.update(loss.item(), model_input.size(0))
        top1.update(acc1.item(), model_input.size(0))

        preds = fusion_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        probs = torch.softmax(fusion_logits, dim=1)
        max_probs = probs.max(dim=1)
        for pred, label, img_path, max_prob in zip(
            preds.cpu().numpy(),
            labels.cpu().numpy(),
            image_paths,
            max_probs.values.cpu().numpy(),
        ):
            if pred != label:
                error_samples.append(
                    {
                        "img_path": img_path,
                        "pred_label": int(pred),
                        "true_label": int(label),
                        "confidence": float(max_prob),
                    }
                )

        pbar.set_postfix({"Loss": f"{losses.avg:.4f}", "Acc": f"{top1.avg:.2f}%"})

    return losses.avg, top1.avg, all_preds, all_labels, error_samples


@torch.no_grad()
def collect_pseudo_labels(model, unlabeled_loader, device, config, class_names):
    model.eval()
    selected_samples = []
    all_confidences: List[float] = []
    all_margins: List[float] = []
    all_entropies: List[float] = []
    per_class_counter: Counter = Counter()

    conf_threshold = float(getattr(config, "PSEUDO_CONF_THRESHOLD", 0.90))
    margin_threshold = float(getattr(config, "PSEUDO_MARGIN_THRESHOLD", 0.25))
    use_entropy = bool(getattr(config, "PSEUDO_USE_ENTROPY_FILTER", False))
    entropy_threshold = float(getattr(config, "PSEUDO_ENTROPY_THRESHOLD", 0.20))
    max_per_class = int(max(0, getattr(config, "PSEUDO_MAX_SAMPLES_PER_CLASS", 0)))

    candidate_samples: List[Dict[str, object]] = []
    pbar = tqdm(unlabeled_loader, desc="[PseudoLabel]")
    for images, rel_paths, image_paths in pbar:
        images = images.to(device, non_blocking=True)
        logits = model(prepare_single_view_input(images))["fusion_logits"]
        probs = torch.softmax(logits, dim=1)
        entropy = compute_entropy_batch(probs)
        top2_probs, top2_indices = probs.topk(k=2, dim=1)
        top1_conf = top2_probs[:, 0]
        top2_conf = top2_probs[:, 1]
        margin = top1_conf - top2_conf

        for idx in range(images.size(0)):
            conf_val = float(top1_conf[idx].item())
            margin_val = float(margin[idx].item())
            entropy_val = float(entropy[idx].item())
            if conf_val <= conf_threshold:
                continue
            if margin_val <= margin_threshold:
                continue
            if use_entropy and entropy_val >= entropy_threshold:
                continue
            pred_label = int(top2_indices[idx, 0].item())
            candidate_samples.append(
                {
                    "image_rel_path": str(rel_paths[idx]),
                    "image_path": str(image_paths[idx]),
                    "pred_label": pred_label,
                    "pred_class": class_names[pred_label],
                    "confidence": conf_val,
                    "margin": margin_val,
                    "entropy": entropy_val,
                }
            )

    candidate_samples.sort(key=lambda item: item["confidence"], reverse=True)
    for sample in candidate_samples:
        pred_label = int(sample["pred_label"])
        if max_per_class > 0 and per_class_counter[pred_label] >= max_per_class:
            continue
        selected_samples.append(sample)
        per_class_counter[pred_label] += 1
        all_confidences.append(float(sample["confidence"]))
        all_margins.append(float(sample["margin"]))
        all_entropies.append(float(sample["entropy"]))

    summary = {
        "total_samples": int(len(unlabeled_loader.dataset)),
        "selected_samples": int(len(selected_samples)),
        "keep_ratio": float(len(selected_samples) / max(1, len(unlabeled_loader.dataset))),
        "thresholds": {
            "confidence": conf_threshold,
            "margin": margin_threshold,
            "use_entropy": use_entropy,
            "entropy": entropy_threshold if use_entropy else None,
        },
        "max_samples_per_class": max_per_class,
        "confidence_distribution": histogram_summary(all_confidences, np.linspace(conf_threshold, 1.0, 11).tolist()),
        "margin_distribution": histogram_summary(all_margins, np.linspace(margin_threshold, 1.0, 11).tolist()),
        "entropy_distribution": histogram_summary(all_entropies, np.linspace(0.0, max(2.0, entropy_threshold, 1.0), 11).tolist()),
        "per_class_counts": {
            class_names[idx]: int(count) for idx, count in sorted(per_class_counter.items())
        },
    }
    return selected_samples, summary


def save_pseudo_label_artifacts(output_dir: Path, pseudo_samples, summary, config) -> Tuple[str, str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pseudo_labels_round1.csv"
    json_path = output_dir / "pseudo_labels_round1.json"
    snapshot_path = output_dir / "config_snapshot.json"

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_rel_path",
                "image_path",
                "pred_label",
                "pred_class",
                "confidence",
                "margin",
                "entropy",
            ],
        )
        writer.writeheader()
        for sample in pseudo_samples:
            writer.writerow(sample)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "samples": pseudo_samples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot_config(config), f, ensure_ascii=False, indent=2)

    return str(csv_path), str(json_path), str(snapshot_path)


def load_saved_pseudo_samples(manifest_path: Path) -> List[Dict[str, object]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"Invalid pseudo manifest format: {manifest_path}")
    return samples


def train_one_epoch_singleview_selftrain(
    model,
    train_loader,
    pseudo_loader,
    criterion,
    optimizer,
    device,
    epoch,
    config,
    helper_real,
    helper_pseudo,
    cam_model=None,
):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    real_loss_meter = AverageMeter()
    pseudo_loss_meter = AverageMeter()
    pseudo_samples_used = 0
    real_samples_seen = 0

    aug_stats = {
        "bg_aug_batches": 0,
        "cutmix_batches": 0,
        "defect_batches": 0,
        "rotate_batches": 0,
        "flip_batches": 0,
        "both_aug_batches": 0,
        "bg_only_batches": 0,
        "cutmix_only_batches": 0,
        "defect_only_batches": 0,
        "rotate_only_batches": 0,
        "flip_only_batches": 0,
        "bg_cutmix_rotate_batches": 0,
        "all_aug_batches": 0,
        "cutmix_defect_conflict_batches": 0,
        "defect_applied_samples": 0,
        "defect_skipped_empty_mask": 0,
        "pair_random": 0,
        "pair_threshold_matched": 0,
        "pair_threshold_fallback_random": 0,
        "pair_no_area_fallback": 0,
    }

    pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train2-Single-SelfTrain]")
    for batch_idx, (images, labels, image_paths) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        real_samples_seen += int(labels.size(0))

        labels_a, labels_b, lam = labels, labels, 1.0
        apply_bg_aug = bool(getattr(config, "BG_AUG_ENABLE", False)) and (
            np.random.rand() < float(getattr(config, "BG_AUG_PROB", 0.0))
        )
        apply_cutmix = bool(getattr(config, "CUTMIX_ENABLE", True)) and (
            np.random.rand() < float(getattr(config, "CUTMIX_PROB", 0.0))
        )
        apply_defect = False
        if not apply_cutmix:
            apply_defect = bool(getattr(config, "DEFECT_ENABLE", False)) and (
                np.random.rand() < float(getattr(config, "DEFECT_PROB", 0.0))
            )
        apply_rotate = bool(getattr(config, "ROTATE_ENABLE", False)) and (
            np.random.rand() < float(getattr(config, "ROTATE_PROB", 0.0))
        )
        apply_flip = bool(getattr(config, "FLIP_ENABLE", False)) and (
            np.random.rand() < float(getattr(config, "FLIP_PROB", 0.0))
        )
        if apply_bg_aug:
            aug_stats["bg_aug_batches"] += 1
        if apply_cutmix:
            aug_stats["cutmix_batches"] += 1
        if apply_defect:
            aug_stats["defect_batches"] += 1
        if apply_rotate:
            aug_stats["rotate_batches"] += 1
        if apply_flip:
            aug_stats["flip_batches"] += 1
        if apply_bg_aug and apply_cutmix:
            aug_stats["both_aug_batches"] += 1
        if apply_bg_aug and not apply_cutmix and not apply_defect and not apply_rotate and not apply_flip:
            aug_stats["bg_only_batches"] += 1
        if apply_cutmix and not apply_bg_aug and not apply_defect and not apply_rotate and not apply_flip:
            aug_stats["cutmix_only_batches"] += 1
        if apply_defect and not apply_bg_aug and not apply_cutmix and not apply_rotate and not apply_flip:
            aug_stats["defect_only_batches"] += 1
        if apply_rotate and not apply_bg_aug and not apply_cutmix and not apply_defect and not apply_flip:
            aug_stats["rotate_only_batches"] += 1
        if apply_flip and not apply_bg_aug and not apply_cutmix and not apply_defect and not apply_rotate:
            aug_stats["flip_only_batches"] += 1
        if apply_bg_aug and apply_cutmix and apply_rotate:
            aug_stats["bg_cutmix_rotate_batches"] += 1
        if apply_bg_aug and (apply_cutmix or apply_defect) and apply_rotate and apply_flip:
            aug_stats["all_aug_batches"] += 1

        if apply_bg_aug or apply_cutmix or apply_defect or apply_rotate or apply_flip:
            images, labels_a, labels_b, lam, batch_stats = apply_singleview_augmentation(
                images,
                labels,
                list(image_paths),
                helper_real,
                model,
                alpha=float(getattr(config, "CUTMIX_ALPHA", 1.0)),
                apply_background_aug=apply_bg_aug,
                apply_defect=apply_defect,
                apply_cutmix=apply_cutmix,
                apply_rotate=apply_rotate,
                apply_flip=apply_flip,
                config=config,
                epoch=epoch,
                batch_idx=batch_idx,
                cam_model=cam_model,
            )
            for key in ("pair_random", "pair_threshold_matched", "pair_threshold_fallback_random", "pair_no_area_fallback"):
                aug_stats[key] += int(batch_stats.get(key, 0))
            aug_stats["defect_applied_samples"] += int(batch_stats.get("defect_applied_samples", 0))
            aug_stats["defect_skipped_empty_mask"] += int(batch_stats.get("defect_skipped_empty_mask", 0))

        model_input = prepare_single_view_input(images)
        optimizer.zero_grad()
        outputs = model(model_input)
        real_loss = _compute_mixed_loss(outputs["fusion_logits"], labels_a, labels_b, lam, criterion, config)
        total_loss = real_loss

        pseudo_loss = None
        if pseudo_iter is not None:
            try:
                pseudo_images, pseudo_labels, pseudo_paths = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = None
                pseudo_images = pseudo_labels = pseudo_paths = None

            if pseudo_iter is not None and pseudo_images is not None:
                pseudo_images = pseudo_images.to(device, non_blocking=True)
                pseudo_labels = pseudo_labels.to(device, non_blocking=True)
                pseudo_labels_a, pseudo_labels_b, pseudo_lam = pseudo_labels, pseudo_labels, 1.0
                if apply_bg_aug or apply_cutmix or apply_defect or apply_rotate or apply_flip:
                    pseudo_images, pseudo_labels_a, pseudo_labels_b, pseudo_lam, _ = apply_singleview_augmentation(
                        pseudo_images,
                        pseudo_labels,
                        list(pseudo_paths),
                        helper_pseudo,
                        model,
                        alpha=float(getattr(config, "CUTMIX_ALPHA", 1.0)),
                        apply_background_aug=apply_bg_aug,
                        apply_defect=apply_defect,
                        apply_cutmix=apply_cutmix,
                        apply_rotate=apply_rotate,
                        apply_flip=apply_flip,
                        config=config,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        cam_model=cam_model,
                    )
                pseudo_outputs = model(prepare_single_view_input(pseudo_images))
                pseudo_loss = _compute_mixed_one_hot_loss(
                    pseudo_outputs["fusion_logits"],
                    pseudo_labels_a,
                    pseudo_labels_b,
                    pseudo_lam,
                )
                total_loss = total_loss + float(getattr(config, "PSEUDO_LOSS_WEIGHT", 0.2)) * pseudo_loss
                pseudo_samples_used += int(pseudo_labels.size(0))
                pseudo_loss_meter.update(pseudo_loss.item(), pseudo_labels.size(0))

        total_loss.backward()
        optimizer.step()

        acc1 = accuracy(outputs["fusion_logits"], labels, topk=(1,))[0]
        losses.update(total_loss.item(), model_input.size(0))
        real_loss_meter.update(real_loss.item(), model_input.size(0))
        top1.update(acc1.item(), model_input.size(0))

        pseudo_ratio = pseudo_samples_used / max(1, real_samples_seen + pseudo_samples_used)
        postfix = {
            "Loss": f"{losses.avg:.4f}",
            "Real": f"{real_loss_meter.avg:.4f}",
            "Acc": f"{top1.avg:.2f}%",
            "Pseudo%": f"{pseudo_ratio * 100.0:.1f}",
        }
        if pseudo_loss_meter.count > 0:
            postfix["Pseudo"] = f"{pseudo_loss_meter.avg:.4f}"
        pbar.set_postfix(postfix)

    stats = {
        "real_loss": real_loss_meter.avg,
        "pseudo_loss": pseudo_loss_meter.avg if pseudo_loss_meter.count > 0 else 0.0,
        "pseudo_samples_in_epoch": int(pseudo_samples_used),
        "pseudo_ratio": float(pseudo_samples_used / max(1, real_samples_seen + pseudo_samples_used)),
        "target_pseudo_ratio": float(getattr(config, "PSEUDO_BATCH_RATIO_LOG", 0.0)),
    }
    stats.update(aug_stats)
    return losses.avg, top1.avg, stats


def save_last_k_epoch_checkpoint(
    model,
    optimizer,
    epoch,
    config,
    best_acc,
    best_macro_f1,
    best_val_loss,
    extra_state=None,
):
    keep_k = int(max(0, getattr(config, "SAVE_LAST_K_EPOCHS", 0)))
    if keep_k <= 0:
        return None

    save_dir = Path(config.CHECKPOINT_DIR) / "last_k"
    save_dir.mkdir(parents=True, exist_ok=True)
    digits = max(3, len(str(int(getattr(config, "NUM_EPOCHS", epoch + 1)))))
    save_path = save_dir / f"epoch_{epoch + 1:0{digits}d}.pth"
    save_checkpoint(
        model,
        optimizer,
        epoch,
        best_acc,
        best_macro_f1,
        best_val_loss,
        str(save_path),
        extra_state=extra_state,
    )

    epoch_files = sorted(save_dir.glob("epoch_*.pth"))
    while len(epoch_files) > keep_k:
        oldest = epoch_files.pop(0)
        oldest.unlink(missing_ok=True)
    return str(save_path)


def main():
    config = Config2()
    if getattr(config, "USE_FREQ_CHANNELS", False):
        raise ValueError("train2.py does not support USE_FREQ_CHANNELS=True")
    train_mode = getattr(config, "TRAIN_MODE", "dual_view")
    single_view_source = getattr(config, "SINGLE_VIEW_SOURCE", "raw")
    self_train_enabled = bool(getattr(config, "SELF_TRAIN_ENABLE", False))
    if train_mode not in {"dual_view", "single_view"}:
        raise ValueError(f"Unknown TRAIN_MODE: {train_mode}")
    if train_mode == "single_view" and single_view_source not in {"raw", "detail"}:
        raise ValueError(f"Unknown SINGLE_VIEW_SOURCE: {single_view_source}")
    if self_train_enabled and train_mode != "single_view":
        raise ValueError("Self-training is only supported for train2 single_view mode.")

    set_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train mode: {train_mode}")
    if train_mode == "single_view":
        print(f"Single-view source: {single_view_source}")

    config.RAW_TRAIN_DIR = str(resolve_existing_path(config.RAW_TRAIN_DIR))
    config.DETAIL_TRAIN_DIR = str(resolve_existing_path(config.DETAIL_TRAIN_DIR))
    config.CLASSNAME_FILE = str(resolve_existing_path(config.CLASSNAME_FILE))
    config.YOLO_CACHE_PATH = str(resolve_existing_path(config.YOLO_CACHE_PATH))
    base_checkpoint_dir = str(resolve_output_path(config.CHECKPOINT_DIR))
    if self_train_enabled:
        config.SELF_TRAIN_SOURCE_DIR = str(resolve_existing_path(config.SELF_TRAIN_SOURCE_DIR))
        config.SELF_TRAIN_YOLO_CACHE_PATH = str(resolve_existing_path(config.SELF_TRAIN_YOLO_CACHE_PATH))
        config.SELF_TRAIN_INIT_CHECKPOINT = str(resolve_existing_path(config.SELF_TRAIN_INIT_CHECKPOINT))
        config.CHECKPOINT_DIR = str(Path(base_checkpoint_dir) / "self_train_round1")
        config.NUM_EPOCHS = int(getattr(config, "SELF_TRAIN_EPOCHS", 8))
        config.LEARNING_RATE = float(getattr(config, "SELF_TRAIN_LEARNING_RATE", config.LEARNING_RATE))
    else:
        config.CHECKPOINT_DIR = base_checkpoint_dir
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    if getattr(config, "PRETRAINED_WEIGHTS_DIR", None):
        config.PRETRAINED_WEIGHTS_DIR = str(resolve_output_path(config.PRETRAINED_WEIGHTS_DIR))

    log_file = os.path.join(config.CHECKPOINT_DIR, f"train2_{get_current_time()}.log")

    print("Loading train2 data...")
    defect_val_loader = None
    if train_mode == "single_view":
        train_loader, val_loader, defect_val_loader, class_names = get_singleview_train2_dataloaders(config)
    else:
        train_loader, val_loader, class_names = get_train2_dataloaders(config)
    pseudo_loader = None
    has_validation = val_loader is not None
    print(f"Number of classes: {len(class_names)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset) if has_validation else 0}")
    if defect_val_loader is not None:
        print(f"Defect val samples: {len(defect_val_loader.dataset)}")
    else:
        print("Defect val samples: 0")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Train mode: {train_mode}\n")
        if train_mode == "single_view":
            f.write(f"Single-view source: {single_view_source}\n")
        f.write(f"Self-train enabled: {self_train_enabled}\n")
        f.write(f"VAL_SPLIT: {getattr(config, 'VAL_SPLIT', 0.2)}\n")
        f.write(f"Validation enabled: {has_validation}\n")
        f.write(f"SAVE_LAST_K_EPOCHS: {int(max(0, getattr(config, 'SAVE_LAST_K_EPOCHS', 0)))}\n")
        if self_train_enabled:
            f.write(f"Self-train source dir: {config.SELF_TRAIN_SOURCE_DIR}\n")
            f.write(f"Self-train init checkpoint: {config.SELF_TRAIN_INIT_CHECKPOINT}\n")
            f.write(f"Self-train learning rate: {config.LEARNING_RATE}\n")

    print("Building train2 model...")
    model = build_train2_model(config)
    if self_train_enabled:
        print(f"Loading self-train init checkpoint: {config.SELF_TRAIN_INIT_CHECKPOINT}")
        _load_model_state_from_checkpoint(model, config.SELF_TRAIN_INIT_CHECKPOINT, device)
    model = model.to(device)

    helper_raw = None
    helper_detail = None
    helper_single = None
    helper_pseudo = None
    defect_cam_model = None
    defect_val_cam_model = None
    if train_mode == "dual_view":
        helper_raw = build_helper_from_config(config, config.RAW_TRAIN_DIR)
        helper_detail = build_helper_from_config(config, config.DETAIL_TRAIN_DIR)
    else:
        single_view_dir = config.RAW_TRAIN_DIR if single_view_source == "raw" else config.DETAIL_TRAIN_DIR
        helper_single = build_helper_from_config(config, single_view_dir)
        if self_train_enabled:
            helper_pseudo = build_helper_with_cache(
                config,
                config.SELF_TRAIN_SOURCE_DIR,
                config.SELF_TRAIN_YOLO_CACHE_PATH,
            )
        if getattr(config, "DEFECT_ENABLE", False):
            defect_cam_model = maybe_build_defect_cam_model(config, device, "DEFECT_CAM_CHECKPOINT_PATH")
        if getattr(config, "DEFECT_VAL_ENABLE", True):
            defect_val_cam_model = maybe_build_defect_cam_model(config, device, "DEFECT_VAL_CAM_CHECKPOINT_PATH")
            if defect_val_cam_model is None:
                defect_val_cam_model = defect_cam_model

    if getattr(config, "USE_MULTI_GPU", False) and torch.cuda.device_count() > 1:
        gpu_ids = getattr(config, "GPU_IDS", list(range(torch.cuda.device_count())))
        print(f"Using DataParallel with GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)

    if config.LOSS_TYPE == "focal":
        alpha = None
        if getattr(config, "USE_CLASS_ALPHA", False):
            targets = torch.tensor(train_loader.dataset.labels)
            class_counts = torch.bincount(targets, minlength=config.NUM_CLASSES).float()
            alpha = 1.0 / (class_counts + 1e-6)
            alpha = alpha / alpha.sum() * config.NUM_CLASSES
        criterion = FocalLoss(
            gamma=config.FOCAL_GAMMA,
            alpha=alpha,
            label_smoothing=config.LABEL_SMOOTHING,
        )
        print(f"Using Focal Loss (gamma={config.FOCAL_GAMMA}, label_smoothing={config.LABEL_SMOOTHING})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        print("Using CrossEntropy Loss")

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    best_acc = 0.0
    best_clean_acc = 0.0
    best_defect_acc = 0.0
    best_mixed_acc = 0.0
    best_macro_f1 = 0.0
    best_val_loss = float("inf")
    start_epoch = 0

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest.pth")
    pseudo_output_dir = Path(config.CHECKPOINT_DIR)
    pseudo_manifest_path = pseudo_output_dir / "pseudo_labels_round1.json"
    if config.RESUME and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, best_acc, best_macro_f1, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path, device)
        resume_checkpoint = torch.load(checkpoint_path, map_location=device)
        best_clean_acc = float(resume_checkpoint.get("best_clean_acc", best_acc))
        best_defect_acc = float(resume_checkpoint.get("best_defect_acc", 0.0))
        best_mixed_acc = float(resume_checkpoint.get("best_mixed_acc", 0.0))
        best_acc = best_clean_acc
        start_epoch += 1
    elif os.path.exists(checkpoint_path):
        print("Checkpoint found but RESUME is disabled. Starting fresh training.")

    pseudo_samples = []
    pseudo_summary = None
    if self_train_enabled:
        if config.RESUME and pseudo_manifest_path.exists():
            pseudo_samples = load_saved_pseudo_samples(pseudo_manifest_path)
            with open(pseudo_manifest_path, "r", encoding="utf-8") as f:
                pseudo_summary = json.load(f).get("summary", {})
            print(f"Loaded {len(pseudo_samples)} pseudo-labeled samples from: {pseudo_manifest_path}")
        else:
            unlabeled_loader = get_singleview_unlabeled_dataloader(config, config.SELF_TRAIN_SOURCE_DIR)
            pseudo_samples, pseudo_summary = collect_pseudo_labels(model, unlabeled_loader, device, config, class_names)
            csv_path, json_path, snapshot_path = save_pseudo_label_artifacts(
                pseudo_output_dir,
                pseudo_samples,
                pseudo_summary,
                config,
            )
            print(f"Saved pseudo-label csv: {csv_path}")
            print(f"Saved pseudo-label json: {json_path}")
            print(f"Saved config snapshot: {snapshot_path}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"Pseudo labels saved to: {csv_path}\n")
                f.write(f"Pseudo summary saved to: {json_path}\n")
                f.write(f"Config snapshot saved to: {snapshot_path}\n")
        if pseudo_summary is not None:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"Pseudo-label summary - total: {pseudo_summary.get('total_samples', 0)}, "
                    f"selected: {pseudo_summary.get('selected_samples', 0)}, "
                    f"keep_ratio: {pseudo_summary.get('keep_ratio', 0.0):.4f}\n"
                )
                f.write(f"Pseudo per-class counts: {pseudo_summary.get('per_class_counts', {})}\n")
        if len(pseudo_samples) == 0:
            print("No pseudo-labeled samples passed the strict filter. Exiting without fine-tuning.")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("No pseudo-labeled samples passed the strict filter. Training aborted.\n")
            return
        pseudo_loader = get_singleview_pseudo_dataloader(config, pseudo_samples)
        print(f"Pseudo-labeled samples: {len(pseudo_loader.dataset)}")

    saved_defect_val_dir = None
    if has_validation and train_mode == "single_view" and defect_val_loader is not None and getattr(config, "DEFECT_VAL_ENABLE", True):
        print("Materializing fixed defect validation images...")
        defect_val_loader, saved_defect_val_dir = materialize_singleview_defect_val_loader(
            defect_val_loader,
            model,
            helper_single,
            config,
            device,
            cam_model=defect_val_cam_model,
        )
        print(f"Saved defect val images to: {saved_defect_val_dir}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Defect val image dir: {saved_defect_val_dir}\n")

    print(f"\nStarting train2 for {config.NUM_EPOCHS} epochs...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        if train_mode == "dual_view":
            train_loss, train_acc, aug_stats = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch + 1,
                config,
                helper_raw,
                helper_detail,
            )
        else:
            if self_train_enabled:
                train_loss, train_acc, aug_stats = train_one_epoch_singleview_selftrain(
                    model,
                    train_loader,
                    pseudo_loader,
                    criterion,
                    optimizer,
                    device,
                    epoch + 1,
                    config,
                    helper_single,
                    helper_pseudo,
                    defect_cam_model,
                )
            else:
                train_loss, train_acc, aug_stats = train_one_epoch_singleview(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    epoch + 1,
                    config,
                    helper_single,
                    defect_cam_model,
                )

        if (
            aug_stats["bg_aug_batches"] > 0
            or aug_stats["cutmix_batches"] > 0
            or aug_stats.get("defect_batches", 0) > 0
            or aug_stats["rotate_batches"] > 0
            or aug_stats["flip_batches"] > 0
        ):
            if train_mode == "single_view":
                print(
                    "Train2 aug stats - "
                    f"bg/cutmix/defect/rotate/flip: "
                    f"{aug_stats['bg_aug_batches']}/"
                    f"{aug_stats['cutmix_batches']}/"
                    f"{aug_stats['defect_batches']}/"
                    f"{aug_stats['rotate_batches']}/"
                    f"{aug_stats['flip_batches']}, "
                    f"only(bg/cutmix/defect/rotate/flip): "
                    f"{aug_stats['bg_only_batches']}/"
                    f"{aug_stats['cutmix_only_batches']}/"
                    f"{aug_stats['defect_only_batches']}/"
                    f"{aug_stats['rotate_only_batches']}/"
                    f"{aug_stats['flip_only_batches']}, "
                    f"combo(bg+cutmix+rotate/all): "
                    f"{aug_stats['bg_cutmix_rotate_batches']}/"
                    f"{aug_stats['all_aug_batches']}, "
                    f"defect(conflict/applied/empty_mask): "
                    f"{aug_stats['cutmix_defect_conflict_batches']}/"
                    f"{aug_stats['defect_applied_samples']}/"
                    f"{aug_stats['defect_skipped_empty_mask']}, "
                    f"pair(rand/matched/fallback/no_area): "
                    f"{aug_stats['pair_random']}/"
                    f"{aug_stats['pair_threshold_matched']}/"
                    f"{aug_stats['pair_threshold_fallback_random']}/"
                    f"{aug_stats['pair_no_area_fallback']}"
                )
            else:
                print(
                    "Train2 aug stats - "
                    f"bg/cutmix/rotate/flip/both/all3/all4: "
                    f"{aug_stats['bg_aug_batches']}/"
                    f"{aug_stats['cutmix_batches']}/"
                    f"{aug_stats['rotate_batches']}/"
                    f"{aug_stats['flip_batches']}/"
                    f"{aug_stats['both_aug_batches']}/"
                    f"{aug_stats['bg_cutmix_rotate_batches']}/"
                    f"{aug_stats['all_aug_batches']}, "
                    f"only(bg/cutmix/rotate/flip): "
                    f"{aug_stats['bg_only_batches']}/"
                    f"{aug_stats['cutmix_only_batches']}/"
                    f"{aug_stats['rotate_only_batches']}/"
                    f"{aug_stats['flip_only_batches']}, "
                    f"pair(rand/matched/fallback/no_area): "
                    f"{aug_stats['pair_random']}/"
                    f"{aug_stats['pair_threshold_matched']}/"
                    f"{aug_stats['pair_threshold_fallback_random']}/"
                    f"{aug_stats['pair_no_area_fallback']}"
                )

        if self_train_enabled and train_mode == "single_view":
            print(
                "Self-train stats - "
                f"real loss: {aug_stats.get('real_loss', 0.0):.4f}, "
                f"pseudo loss: {aug_stats.get('pseudo_loss', 0.0):.4f}, "
                f"pseudo ratio: {aug_stats.get('pseudo_ratio', 0.0) * 100.0:.2f}%, "
                f"pseudo samples/epoch: {aug_stats.get('pseudo_samples_in_epoch', 0)}"
            )

        if has_validation:
            if train_mode == "dual_view":
                val_loss, val_acc, all_preds, all_labels, error_samples = validate(model, val_loader, criterion, device, config)
                clean_val_loss = val_loss
                clean_val_acc = val_acc
                defect_val_loss = None
                defect_val_acc = None
                mixed_val_acc = None
            else:
                clean_val_loss, clean_val_acc, all_preds, all_labels, error_samples = validate_singleview(
                    model,
                    val_loader,
                    criterion,
                    device,
                )
                if getattr(config, "DEFECT_VAL_ENABLE", True) and defect_val_loader is not None:
                    defect_val_loss, defect_val_acc, defect_preds, defect_labels, _ = validate_singleview(
                        model,
                        defect_val_loader,
                        criterion,
                        device,
                        cam_model=defect_val_cam_model,
                    )
                    mixed_correct = 0
                    mixed_total = len(all_labels) + len(defect_labels)
                    if mixed_total > 0:
                        mixed_correct = int(sum(int(p == y) for p, y in zip(all_preds, all_labels)))
                        mixed_correct += int(sum(int(p == y) for p, y in zip(defect_preds, defect_labels)))
                        mixed_val_acc = (100.0 * mixed_correct) / mixed_total
                    else:
                        mixed_val_acc = 0.0
                else:
                    defect_val_loss = None
                    defect_val_acc = None
                    mixed_val_acc = clean_val_acc

                val_loss = clean_val_loss
                val_acc = clean_val_acc

            metrics_dict = compute_class_metrics(
                all_preds,
                all_labels,
                num_classes=config.NUM_CLASSES,
                class_names=class_names,
            )
            print_class_metrics(metrics_dict, class_names)
            print_confusion_matrix(metrics_dict["confusion_matrix"], class_names)
            val_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            if train_mode == "single_view":
                if defect_val_acc is not None:
                    print(
                        f"Validation - clean acc: {clean_val_acc:.2f}%, "
                        f"defect acc: {defect_val_acc:.2f}%, "
                        f"mixed acc: {mixed_val_acc:.2f}%"
                    )
                else:
                    print(f"Validation - clean acc: {clean_val_acc:.2f}%")

            if getattr(config, "SAVE_ERROR_SAMPLES", False) and error_samples:
                save_error_samples(epoch + 1, error_samples, config.ERROR_SAMPLES_DIR, class_names)
        else:
            val_loss = None
            val_acc = None
            clean_val_loss = None
            clean_val_acc = None
            defect_val_loss = None
            defect_val_acc = None
            mixed_val_acc = None
            val_macro_f1 = None
            print("Validation skipped because VAL_SPLIT=0")

        scheduler.step()
        log_training(epoch + 1, config.NUM_EPOCHS, train_loss, train_acc, val_loss, val_acc, log_file)
        if not has_validation:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("Validation skipped because VAL_SPLIT=0\n")
        if train_mode == "single_view":
            with open(log_file, "a", encoding="utf-8") as f:
                if self_train_enabled:
                    f.write(
                        "Self-Train Metrics - "
                        f"Real Loss: {aug_stats.get('real_loss', 0.0):.4f}, "
                        f"Pseudo Loss: {aug_stats.get('pseudo_loss', 0.0):.4f}, "
                        f"Pseudo Ratio: {aug_stats.get('pseudo_ratio', 0.0) * 100.0:.2f}%, "
                        f"Pseudo Samples: {aug_stats.get('pseudo_samples_in_epoch', 0)}, "
                        f"Target Pseudo Ratio: {aug_stats.get('target_pseudo_ratio', 0.0) * 100.0:.2f}%\n"
                    )
                if not has_validation:
                    pass
                elif defect_val_acc is not None:
                    f.write(
                        "Validation Metrics - "
                        f"Clean Val Loss: {clean_val_loss:.4f}, "
                        f"Clean Val Acc: {clean_val_acc:.2f}%, "
                        f"Defect Val Loss: {defect_val_loss:.4f}, "
                        f"Defect Val Acc: {defect_val_acc:.2f}%, "
                        f"Mixed Val Acc: {mixed_val_acc:.2f}%\n"
                    )
                else:
                    f.write(
                        "Validation Metrics - "
                        f"Clean Val Loss: {clean_val_loss:.4f}, "
                        f"Clean Val Acc: {clean_val_acc:.2f}%\n"
                    )

        is_best_acc = has_validation and val_acc > best_acc
        is_best_macro_f1 = has_validation and val_macro_f1 > best_macro_f1
        is_best_loss = has_validation and val_loss < best_val_loss
        is_best_clean_acc = has_validation and clean_val_acc > best_clean_acc
        is_best_defect_acc = has_validation and defect_val_acc is not None and defect_val_acc > best_defect_acc
        is_best_mixed_acc = has_validation and mixed_val_acc is not None and mixed_val_acc > best_mixed_acc

        checkpoint_extra_state = {
            "best_clean_acc": best_clean_acc,
            "best_defect_acc": best_defect_acc,
            "best_mixed_acc": best_mixed_acc,
        }

        if is_best_acc:
            best_acc = val_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_acc,
                best_macro_f1,
                best_val_loss,
                os.path.join(config.CHECKPOINT_DIR, "best_acc.pth"),
                extra_state=checkpoint_extra_state,
            )
            print(f"Best accuracy updated: {best_acc:.2f}%")

        if is_best_macro_f1:
            best_macro_f1 = val_macro_f1
            checkpoint_extra_state["best_macro_f1"] = best_macro_f1
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_acc,
                best_macro_f1,
                best_val_loss,
                os.path.join(config.CHECKPOINT_DIR, "best_macro_f1.pth"),
                extra_state=checkpoint_extra_state,
            )
            print(f"Best macro-F1 updated: {best_macro_f1:.4f}")

        if is_best_loss:
            best_val_loss = val_loss
            checkpoint_extra_state["best_val_loss"] = best_val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_acc,
                best_macro_f1,
                best_val_loss,
                os.path.join(config.CHECKPOINT_DIR, "best_loss.pth"),
                extra_state=checkpoint_extra_state,
            )
            print(f"Best val loss updated: {best_val_loss:.4f}")

        if is_best_clean_acc:
            best_clean_acc = clean_val_acc
            best_acc = best_clean_acc
            checkpoint_extra_state["best_clean_acc"] = best_clean_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_clean_acc,
                best_macro_f1,
                best_val_loss,
                os.path.join(config.CHECKPOINT_DIR, "best_clean_acc.pth"),
                extra_state=checkpoint_extra_state,
            )
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_clean_acc,
                best_macro_f1,
                best_val_loss,
                os.path.join(config.CHECKPOINT_DIR, "best_acc.pth"),
                verbose=False,
                extra_state=checkpoint_extra_state,
            )
            print(f"Best clean accuracy updated: {best_clean_acc:.2f}%")

        if is_best_defect_acc:
            best_defect_acc = float(defect_val_acc)
            checkpoint_extra_state["best_defect_acc"] = best_defect_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_clean_acc,
                best_macro_f1,
                best_val_loss,
                os.path.join(config.CHECKPOINT_DIR, "best_defect_acc.pth"),
                extra_state=checkpoint_extra_state,
            )
            print(f"Best defect accuracy updated: {best_defect_acc:.2f}%")

        if is_best_mixed_acc:
            best_mixed_acc = float(mixed_val_acc)
            checkpoint_extra_state["best_mixed_acc"] = best_mixed_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_clean_acc,
                best_macro_f1,
                best_val_loss,
                os.path.join(config.CHECKPOINT_DIR, "best_mixed_acc.pth"),
                extra_state=checkpoint_extra_state,
            )
            print(f"Best mixed accuracy updated: {best_mixed_acc:.2f}%")

        checkpoint_extra_state = {
            "best_clean_acc": best_clean_acc,
            "best_defect_acc": best_defect_acc,
            "best_mixed_acc": best_mixed_acc,
        }
        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_clean_acc,
            best_macro_f1,
            best_val_loss,
            os.path.join(config.CHECKPOINT_DIR, "latest.pth"),
            extra_state=checkpoint_extra_state,
        )
        rolling_checkpoint_path = save_last_k_epoch_checkpoint(
            model,
            optimizer,
            epoch,
            config,
            best_clean_acc,
            best_macro_f1,
            best_val_loss,
            extra_state=checkpoint_extra_state,
        )

        if has_validation:
            print(
                f"Current - Best Clean Acc: {best_clean_acc:.2f}%, "
                f"Best Defect Acc: {best_defect_acc:.2f}%, "
                f"Best Mixed Acc: {best_mixed_acc:.2f}%, "
                f"Best Macro-F1: {best_macro_f1:.4f}, "
                f"Best Val Loss: {best_val_loss:.4f}"
            )
        else:
            print("Current - Validation disabled; best_* checkpoints not updated.")
        if train_mode == "single_view":
            with open(log_file, "a", encoding="utf-8") as f:
                if has_validation:
                    f.write(
                        "Best Metrics - "
                        f"Best Clean Acc: {best_clean_acc:.2f}%, "
                        f"Best Defect Acc: {best_defect_acc:.2f}%, "
                        f"Best Mixed Acc: {best_mixed_acc:.2f}%, "
                        f"Best Macro-F1: {best_macro_f1:.4f}, "
                        f"Best Val Loss: {best_val_loss:.4f}\n"
                    )
                else:
                    f.write("Best Metrics - Validation disabled; best_* checkpoints not updated.\n")
                if rolling_checkpoint_path is not None:
                    f.write(f"Saved rolling epoch checkpoint: {rolling_checkpoint_path}\n")

        if rolling_checkpoint_path is not None:
            print(f"Saved rolling epoch checkpoint: {rolling_checkpoint_path}")

    print(f"\n{'=' * 50}")
    print("Train2 completed!")
    if has_validation:
        print(f"Best clean accuracy: {best_clean_acc:.2f}%")
        print(f"Best defect accuracy: {best_defect_acc:.2f}%")
        print(f"Best mixed accuracy: {best_mixed_acc:.2f}%")
        print(f"Best macro-F1: {best_macro_f1:.4f}")
        print(f"Best val loss: {best_val_loss:.4f}")
    else:
        print("Validation was disabled; best_* checkpoints were not updated.")
    print(f"Model saved to: {config.CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
