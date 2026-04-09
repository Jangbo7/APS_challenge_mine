import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import make_grid, save_image

from augmentation import occamix_bgfill_data
from config import Config
from model import build_model
from utils import load_checkpoint, set_seed
from yolo_cutmix import YoloCutMixHelper

try:
    from skimage import segmentation

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VALID_DAMAGE_MODES = {"local_black", "gaussian_noise", "yolo_bg_fill"}


def resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates = [(Path.cwd() / path).resolve(), (Path(__file__).resolve().parent / path).resolve()]
    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def list_classification_images(train_dir: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not train_dir.is_dir():
        raise NotADirectoryError(f"Train directory is not a directory: {train_dir}")

    class_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class subdirectories found in: {train_dir}")

    class_names = [p.name for p in class_dirs]
    items: List[Tuple[Path, int]] = []
    for class_idx, class_dir in enumerate(class_dirs):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                items.append((image_path, class_idx))

    if not items:
        raise RuntimeError(f"No images found under class subdirectories in: {train_dir}")
    return items, class_names


def load_image_tensor(img_path: Path, image_size: int) -> torch.Tensor:
    with Image.open(img_path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.BILINEAR)
    return TF.to_tensor(image)


def make_model_config(base_config: Config, args: argparse.Namespace) -> SimpleNamespace:
    cfg = SimpleNamespace()
    for name in dir(base_config):
        if name.startswith("_"):
            continue
        value = getattr(base_config, name)
        if callable(value):
            continue
        setattr(cfg, name, value)

    cfg.MODEL_TYPE = args.model_type
    cfg.NUM_CLASSES = args.num_classes
    cfg.IMAGE_SIZE = args.image_size
    return cfg


def generate_occamix_masks(
    images: torch.Tensor,
    labels: torch.Tensor,
    model,
    n_top: int,
    n_seg_min: int,
    n_seg_max: int,
    compactness: float,
):
    if not SKIMAGE_AVAILABLE:
        raise ImportError("OcCaMix visualization requires scikit-image. Install with: pip install scikit-image")

    bsz, _, h_img, w_img = images.shape
    rand_idx = torch.randperm(bsz, device=images.device)
    labels_a = labels
    labels_b = labels[rand_idx]

    model_ref = model.module if hasattr(model, "module") else model
    was_training = model.training
    model.eval()

    with torch.no_grad():
        feat_map = model_ref.get_spatial_feature_map(images[rand_idx])
        cls_weight = model_ref.get_classifier_weight().detach()

        if feat_map.ndim != 4:
            raise RuntimeError(f"OcCaMix expects 4D feature map [B,C,H,W], got {tuple(feat_map.shape)}")

        cam_all = torch.einsum("kc,bchw->bkhw", cls_weight.to(feat_map.device), feat_map)
        cam_all = torch.relu(cam_all)
        _, _, h_fmap, w_fmap = cam_all.shape
        eval_train_map = cam_all.amax(dim=1).view(bsz, -1)

    if was_training:
        model.train()

    n_top = max(1, min(int(n_top), eval_train_map.size(1)))
    _, map_topn_idx = torch.topk(eval_train_map, n_top, dim=1, largest=True)
    map_topn_row = (map_topn_idx // w_fmap).cpu().numpy()
    map_topn_col = (map_topn_idx % w_fmap).cpu().numpy()

    mask_batch: List[torch.Tensor] = []
    lam_batch: List[float] = []

    for i in range(bsz):
        img_seg = images[rand_idx[i]].detach().permute(1, 2, 0).cpu().numpy()
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

        mask_batch.append(torch.from_numpy(selected_mask))
        damage_ratio = float(selected_mask.sum()) / float(h_img * w_img)
        lam_batch.append(1.0 - damage_ratio)

    masks = torch.stack(mask_batch, dim=0).to(device=images.device, dtype=torch.bool)
    lam = torch.tensor(lam_batch, dtype=images.dtype, device=images.device)
    return masks, lam, labels_a, labels_b


def apply_damage(images: torch.Tensor, masks: torch.Tensor, mode: str, noise_std: float) -> torch.Tensor:
    if mode not in VALID_DAMAGE_MODES:
        raise ValueError(f"Unsupported mode: {mode}. Expected one of {sorted(VALID_DAMAGE_MODES)}")
    if mode == "yolo_bg_fill":
        raise ValueError("mode='yolo_bg_fill' must be executed through occamix_bgfill_data")

    damaged = images.clone()
    mask_expanded = masks.unsqueeze(1)

    if mode == "local_black":
        damaged = damaged.masked_fill(mask_expanded, 0.0)
    elif mode == "gaussian_noise":
        noise = torch.randn_like(damaged) * float(noise_std)
        damaged = torch.where(mask_expanded, damaged + noise, damaged)
        damaged = torch.clamp(damaged, 0.0, 1.0)

    return damaged


def mask_to_rgb(mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(dtype=torch.float32)
    return mask_float.unsqueeze(0).repeat(3, 1, 1)


def make_overlay(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.45) -> torch.Tensor:
    overlay = image.clone()
    red = torch.tensor([1.0, 0.0, 0.0], dtype=image.dtype, device=image.device).view(3, 1, 1)
    mask_expanded = mask.unsqueeze(0).to(dtype=image.dtype)
    overlay = torch.where(
        mask_expanded > 0,
        torch.clamp((1.0 - alpha) * overlay + alpha * red, 0.0, 1.0),
        overlay,
    )
    return overlay


def save_quad_panel(
    original: torch.Tensor,
    mask: torch.Tensor,
    overlay: torch.Tensor,
    damaged: torch.Tensor,
    output_path: Path,
) -> None:
    panel = make_grid(
        [
            original.detach().cpu(),
            mask_to_rgb(mask.detach().cpu()),
            overlay.detach().cpu(),
            damaged.detach().cpu(),
        ],
        nrow=4,
        padding=4,
    )
    save_image(panel, str(output_path))


def build_parser() -> argparse.ArgumentParser:
    base_config = Config()
    # default_checkpoint = Path(base_config.CHECKPOINT_DIR) / "latest.pth"
    default_checkpoint =  "eff/best_loss_f1.pth"

    parser = argparse.ArgumentParser(description="Visualize OcCaMix masks with local damage modes.")
    parser.add_argument("--train-dir", type=str, default=base_config.TRAIN_DIR)
    parser.add_argument("--checkpoint-path", type=str, default=str(default_checkpoint))
    parser.add_argument("--out-dir", type=str, default="eff/occamix_damage_preview")
    parser.add_argument("--model-type", type=str, default=base_config.MODEL_TYPE)
    parser.add_argument("--num-classes", type=int, default=base_config.NUM_CLASSES)
    parser.add_argument("--image-size", type=int, default=base_config.IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--device", type=str, default=base_config.DEVICE)
    parser.add_argument("--seed", type=int, default=base_config.SEED)
    parser.add_argument("--mode", type=str, default="local_black", choices=sorted(VALID_DAMAGE_MODES))
    parser.add_argument("--noise-std", type=float, default=0.12)
    parser.add_argument("--n-top", type=int, default=1)
    parser.add_argument("--n-seg-min", type=int, default=180)
    parser.add_argument("--n-seg-max", type=int, default=220)
    parser.add_argument("--compactness", type=float, default=1)
    parser.add_argument("--yolo-cache-path", type=str, default=getattr(base_config, "OCCAMIX_BG_CACHE_PATH", "eff/yolo_boxes_cache.json"))
    parser.add_argument("--yolo-key-mode", type=str, default=getattr(base_config, "OCCAMIX_BG_KEY_MODE", "relative_to_train_dir"))
    parser.add_argument("--enable-center-shift", type=int, default=int(getattr(base_config, "OCCAMIX_BG_ENABLE_RECENTER_SHIFT", False)), choices=[0, 1])
    parser.add_argument("--bg-aug-enable", type=int, default=int(getattr(base_config, "OCCAMIX_BG_AUG_ENABLE", True)), choices=[0, 1])
    parser.add_argument("--bg-bleed-into-box-ratio", type=float, default=getattr(base_config, "OCCAMIX_BG_BLEED_INTO_BOX_RATIO", 0.07))
    parser.add_argument("--bg-black-dot-prob", type=float, default=getattr(base_config, "OCCAMIX_BG_BLACK_DOT_PROB", 0.0005))
    parser.add_argument("--bg-black-dot-size-min", type=int, default=getattr(base_config, "OCCAMIX_BG_BLACK_DOT_SIZE_MIN", 2))
    parser.add_argument("--bg-black-dot-size-max", type=int, default=getattr(base_config, "OCCAMIX_BG_BLACK_DOT_SIZE_MAX", 4))
    parser.add_argument("--bg-blur-sigma-min", type=float, default=getattr(base_config, "OCCAMIX_BG_BLUR_SIGMA_MIN", 1.2))
    parser.add_argument("--bg-blur-sigma-max", type=float, default=getattr(base_config, "OCCAMIX_BG_BLUR_SIGMA_MAX", 2.8))
    parser.add_argument("--bg-brightness", type=float, default=getattr(base_config, "OCCAMIX_BG_BRIGHTNESS", 0.45))
    parser.add_argument("--bg-contrast", type=float, default=getattr(base_config, "OCCAMIX_BG_CONTRAST", 0.55))
    parser.add_argument("--bg-saturation", type=float, default=getattr(base_config, "OCCAMIX_BG_SATURATION", 0.50))
    parser.add_argument("--bg-hue", type=float, default=getattr(base_config, "OCCAMIX_BG_HUE", 0.3))
    parser.add_argument("--target-expand-ratio", type=float, default=getattr(base_config, "OCCAMIX_BG_TARGET_EXPAND_RATIO", 0.0))
    parser.add_argument("--fill-blur-sigma", type=float, default=getattr(base_config, "OCCAMIX_BG_FILL_BLUR_SIGMA", 0.0))
    parser.add_argument("--n-show-per-batch", type=int, default=4)
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()

    if args.n_seg_min > args.n_seg_max:
        raise ValueError(f"--n-seg-min must be <= --n-seg-max, got {args.n_seg_min} > {args.n_seg_max}")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for OcCaMix pairing")
    if args.n_show_per_batch < 1:
        raise ValueError("--n-show-per-batch must be >= 1")
    if args.mode == "gaussian_noise" and args.noise_std <= 0:
        raise ValueError("--noise-std must be > 0 when --mode gaussian_noise")
    if not SKIMAGE_AVAILABLE:
        raise ImportError("OcCaMix visualization requires scikit-image. Install with: pip install scikit-image")

    set_seed(args.seed)

    train_dir = resolve_existing_path(args.train_dir)
    checkpoint_path = resolve_existing_path(args.checkpoint_path)
    out_dir = resolve_output_path(args.out_dir)
    yolo_cache_path = resolve_existing_path(args.yolo_cache_path) if args.mode == "yolo_bg_fill" else None

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.mode == "yolo_bg_fill" and (yolo_cache_path is None or not yolo_cache_path.exists()):
        raise FileNotFoundError(f"YOLO cache not found: {args.yolo_cache_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "panels"
    images_dir.mkdir(parents=True, exist_ok=True)

    items, class_names = list_classification_images(train_dir)
    if len(items) < args.batch_size:
        raise RuntimeError(f"Not enough images under {train_dir}: {len(items)} < batch_size={args.batch_size}")

    random.shuffle(items)

    runtime_config = make_model_config(Config(), args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Train dir: {train_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {out_dir}")
    print(f"Mode: {args.mode}")
    if args.mode == "yolo_bg_fill":
        print(f"YOLO cache: {yolo_cache_path}")

    model = build_model(runtime_config, pretrained=False)
    model = model.to(device)
    load_checkpoint(model, None, str(checkpoint_path), device=device, verbose=True)
    model.eval()

    ptr = 0
    summary: Dict[str, object] = {
        "train_dir": str(train_dir),
        "checkpoint_path": str(checkpoint_path),
        "out_dir": str(out_dir),
        "device": str(device),
        "mode": args.mode,
        "class_count": len(class_names),
        "num_batches_requested": int(args.num_batches),
        "num_batches_processed": 0,
        "samples_processed": 0,
        "samples_saved": 0,
        "mode_counts": {args.mode: 0},
        "lam_mean": 0.0,
        "damage_ratio_mean": 0.0,
        "valid_box_samples": 0,
        "bg_aug_applied_samples": 0,
        "bg_pool_ratio_mean": 0.0,
        "target_intersection_ratio_mean": 0.0,
        "skipped_no_box": 0,
        "skipped_no_background": 0,
        "skipped_empty_target": 0,
        "args": vars(args),
    }

    lam_values: List[float] = []
    damage_ratios: List[float] = []
    bg_pool_ratios: List[float] = []
    target_intersection_ratios: List[float] = []

    yolo_helper = None
    if args.mode == "yolo_bg_fill":
        yolo_helper = YoloCutMixHelper(
            cache_path=str(yolo_cache_path),
            train_dir=str(train_dir),
            key_mode=args.yolo_key_mode,
            fallback_mode="skip",
            min_box_area_ratio=float(getattr(Config(), "OCCAMIX_BG_MIN_BOX_AREA_RATIO", 0.0)),
            max_box_area_ratio=float(getattr(Config(), "OCCAMIX_BG_MAX_BOX_AREA_RATIO", 0.8)),
            sector_center_jitter_ratio=0.0,
            enable_recenter_shift=bool(args.enable_center_shift),
            center_tolerance_ratio=float(getattr(Config(), "OCCAMIX_BG_CENTER_TOLERANCE_RATIO", 0.10)),
            debug_log=False,
            pair_use_area_match=False,
            pair_random_prob=1.0,
            pair_area_ratio_min=0.0,
            pair_area_ratio_max=1.0,
            pair_area_ratio_min_target=0.0,
            pair_area_ratio_max_target=1.0,
            pair_schedule_start_ratio=0.0,
            pair_schedule_end_ratio=0.0,
        )

    for batch_idx in range(args.num_batches):
        batch_items = items[ptr : ptr + args.batch_size]
        if len(batch_items) < args.batch_size:
            break
        ptr += args.batch_size

        img_paths = [p for p, _ in batch_items]
        labels = torch.tensor([label for _, label in batch_items], dtype=torch.long, device=device)
        images = torch.stack([load_image_tensor(p, args.image_size) for p in img_paths], dim=0).to(device)

        panel_base_images = images
        if args.mode == "yolo_bg_fill":
            damaged, labels_a, labels_b, lam, details = occamix_bgfill_data(
                images=images,
                labels=labels,
                img_paths=[str(p) for p in img_paths],
                model=model,
                yolo_helper=yolo_helper,
                n_top=args.n_top,
                n_seg_min=args.n_seg_min,
                n_seg_max=args.n_seg_max,
                compactness=args.compactness,
                bg_aug_enable=bool(args.bg_aug_enable),
                bg_bleed_into_box_ratio=float(max(0.0, args.bg_bleed_into_box_ratio)),
                bg_black_dot_prob=float(max(0.0, args.bg_black_dot_prob)),
                bg_black_dot_size_min=int(max(1, args.bg_black_dot_size_min)),
                bg_black_dot_size_max=int(max(1, args.bg_black_dot_size_max)),
                bg_blur_sigma_min=float(max(0.0, args.bg_blur_sigma_min)),
                bg_blur_sigma_max=float(max(0.0, args.bg_blur_sigma_max)),
                bg_brightness=float(max(0.0, args.bg_brightness)),
                bg_contrast=float(max(0.0, args.bg_contrast)),
                bg_saturation=float(max(0.0, args.bg_saturation)),
                bg_hue=float(max(0.0, min(0.5, args.bg_hue))),
                target_expand_ratio=float(max(0.0, args.target_expand_ratio)),
                fill_blur_sigma=float(max(0.0, args.fill_blur_sigma)),
                return_details=True,
            )
            masks = details["target_masks"]
            panel_base_images = details["pre_fill_images"]
            batch_stats = details["stats"]
            summary["valid_box_samples"] = int(summary["valid_box_samples"]) + int(batch_stats["valid_box_samples"])
            summary["bg_aug_applied_samples"] = int(summary["bg_aug_applied_samples"]) + int(batch_stats["bg_aug_applied_samples"])
            summary["skipped_no_box"] = int(summary["skipped_no_box"]) + int(batch_stats["skipped_no_box"])
            summary["skipped_no_background"] = int(summary["skipped_no_background"]) + int(batch_stats["skipped_no_background"])
            summary["skipped_empty_target"] = int(summary["skipped_empty_target"]) + int(batch_stats["skipped_empty_target"])
            valid_box_den = max(1, int(batch_stats["valid_box_samples"]))
            bg_pool_ratios.append(float(batch_stats["bg_pool_ratio_sum"]) / valid_box_den)
            target_intersection_ratios.append(float(batch_stats["target_intersection_ratio_sum"]) / valid_box_den)
        else:
            masks, lam, labels_a, labels_b = generate_occamix_masks(
                images=images,
                labels=labels,
                model=model,
                n_top=args.n_top,
                n_seg_min=args.n_seg_min,
                n_seg_max=args.n_seg_max,
                compactness=args.compactness,
            )
            damaged = apply_damage(images=images, masks=masks, mode=args.mode, noise_std=args.noise_std)

        save_count = min(args.n_show_per_batch, images.size(0))
        for sample_idx in range(save_count):
            mask = masks[sample_idx]
            overlay = make_overlay(panel_base_images[sample_idx], mask)
            damage_ratio = float(mask.float().mean().item())
            filename = (
                f"b{batch_idx:03d}_i{sample_idx:03d}"
                f"_la{int(labels_a[sample_idx].item())}"
                f"_lb{int(labels_b[sample_idx].item())}"
                f"_{args.mode}"
                f"_lam{float(lam[sample_idx].item()):.4f}"
                f"_damage{damage_ratio:.4f}.png"
            )
            save_quad_panel(
                original=panel_base_images[sample_idx],
                mask=mask,
                overlay=overlay,
                damaged=damaged[sample_idx],
                output_path=images_dir / filename,
            )

        batch_lam = lam.detach().cpu().numpy().tolist()
        batch_damage = [float(masks[i].float().mean().item()) for i in range(masks.size(0))]
        lam_values.extend(float(v) for v in batch_lam)
        damage_ratios.extend(batch_damage)

        summary["num_batches_processed"] = int(summary["num_batches_processed"]) + 1
        summary["samples_processed"] = int(summary["samples_processed"]) + int(images.size(0))
        summary["samples_saved"] = int(summary["samples_saved"]) + int(save_count)
        summary["mode_counts"][args.mode] = int(summary["mode_counts"][args.mode]) + int(images.size(0))

        print(
            f"[Batch {batch_idx}] processed={images.size(0)} "
            f"saved={save_count} lam_mean={float(lam.mean().item()):.4f} "
            f"damage_ratio_mean={float(np.mean(batch_damage)):.4f}"
        )

    if int(summary["num_batches_processed"]) == 0:
        raise RuntimeError("No batches were processed. Increase available images or reduce batch size/num_batches.")

    summary["lam_mean"] = float(np.mean(lam_values)) if lam_values else 0.0
    summary["damage_ratio_mean"] = float(np.mean(damage_ratios)) if damage_ratios else 0.0
    summary["bg_pool_ratio_mean"] = float(np.mean(bg_pool_ratios)) if bg_pool_ratios else 0.0
    summary["target_intersection_ratio_mean"] = float(np.mean(target_intersection_ratios)) if target_intersection_ratios else 0.0

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved outputs:")
    print(images_dir)
    print(out_dir / "summary.json")


if __name__ == "__main__":
    main()
