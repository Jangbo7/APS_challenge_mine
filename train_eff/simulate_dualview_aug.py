import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import make_grid, save_image

from yolo_cutmix import YoloCutMixHelper


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Default parameters are intentionally defined here, not loaded from config.py/config4yolo.py.
DEFAULT_RAW_TRAIN_DIR = r"data\APS_dataset\train"
DEFAULT_DETAIL_TRAIN_DIR = r"data\APS_dataset_yolo224\train"
DEFAULT_CACHE_PATH = r"eff\yolo_boxes_cache.json"
DEFAULT_OUT_DIR = r"eff\dualview_cutmix_preview"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_BATCHES = 4
DEFAULT_N_SHOW_PER_BATCH = 7
DEFAULT_DEVICE = "cuda"
DEFAULT_SEED = 17
DEFAULT_CUTMIX_ENABLE = 1
DEFAULT_CUTMIX_PROB = 1.0
DEFAULT_ALPHA = 1.4
DEFAULT_ENABLE_CENTER_SHIFT = 1
DEFAULT_KEY_MODE = "relative_to_train_dir"
DEFAULT_MIN_BOX_AREA_RATIO = 0.0
DEFAULT_MAX_BOX_AREA_RATIO = 0.8
DEFAULT_SECTOR_CENTER_JITTER_RATIO = 0
DEFAULT_CENTER_TOLERANCE_RATIO = 0
DEFAULT_PAIR_USE_AREA_MATCH = 1
DEFAULT_PAIR_RANDOM_PROB = 0.1
DEFAULT_PAIR_AREA_RATIO_MIN = 0.4
DEFAULT_PAIR_AREA_RATIO_MAX = 2.5
DEFAULT_BG_AUG_ENABLE = 1
DEFAULT_BG_AUG_PROB = 1.0
DEFAULT_BG_AUG_BLEED_INTO_BOX_RATIO = 0.07
DEFAULT_BG_AUG_BLACK_DOT_PROB = 0.0005
DEFAULT_BG_AUG_BLACK_DOT_SIZE_MIN = 2
DEFAULT_BG_AUG_BLACK_DOT_SIZE_MAX = 4
DEFAULT_BG_AUG_BLUR_SIGMA_MIN = 1.2
DEFAULT_BG_AUG_BLUR_SIGMA_MAX = 2.8
DEFAULT_BG_AUG_BRIGHTNESS = 0.45
DEFAULT_BG_AUG_CONTRAST = 0.55
DEFAULT_BG_AUG_SATURATION = 0.50
DEFAULT_BG_AUG_HUE = 0.3


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


def list_view_images(root: Path) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"View directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"View directory is not a directory: {root}")

    image_map: Dict[str, Path] = {}
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = str(img_path.relative_to(root)).replace("\\", "/")
            image_map[rel] = img_path
    if not image_map:
        raise RuntimeError(f"No images found under: {root}")
    return image_map


def build_paired_items(raw_root: Path, detail_root: Path) -> List[Tuple[str, Path, Path, int]]:
    raw_map = list_view_images(raw_root)
    detail_map = list_view_images(detail_root)

    raw_keys = set(raw_map.keys())
    detail_keys = set(detail_map.keys())
    missing_detail = sorted(raw_keys - detail_keys)
    missing_raw = sorted(detail_keys - raw_keys)
    if missing_detail:
        raise RuntimeError(f"Detail view is missing {len(missing_detail)} files. First missing pair: {missing_detail[0]}")
    if missing_raw:
        raise RuntimeError(f"Raw view is missing {len(missing_raw)} files. First extra detail pair: {missing_raw[0]}")

    class_names = sorted({Path(k).parts[0] for k in raw_map.keys()})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    return [(rel, raw_map[rel], detail_map[rel], class_to_idx[Path(rel).parts[0]]) for rel in sorted(raw_map.keys())]


def load_image_tensor(img_path: Path, image_size: int) -> torch.Tensor:
    with Image.open(img_path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return TF.to_tensor(image)


def build_helper(cache_path: str, train_dir: str, enable_center_shift: bool) -> YoloCutMixHelper:
    return YoloCutMixHelper(
        cache_path=cache_path,
        train_dir=train_dir,
        key_mode=DEFAULT_KEY_MODE,
        fallback_mode="skip",
        min_box_area_ratio=DEFAULT_MIN_BOX_AREA_RATIO,
        max_box_area_ratio=DEFAULT_MAX_BOX_AREA_RATIO,
        sector_center_jitter_ratio=DEFAULT_SECTOR_CENTER_JITTER_RATIO,
        enable_recenter_shift=enable_center_shift,
        center_tolerance_ratio=DEFAULT_CENTER_TOLERANCE_RATIO,
        debug_log=False,
        pair_use_area_match=bool(DEFAULT_PAIR_USE_AREA_MATCH),
        pair_random_prob=DEFAULT_PAIR_RANDOM_PROB,
        pair_area_ratio_min=DEFAULT_PAIR_AREA_RATIO_MIN,
        pair_area_ratio_max=DEFAULT_PAIR_AREA_RATIO_MAX,
        pair_area_ratio_min_target=DEFAULT_PAIR_AREA_RATIO_MIN,
        pair_area_ratio_max_target=DEFAULT_PAIR_AREA_RATIO_MAX,
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
    return mixed, info


def choose_runtime_device(device_arg: str) -> torch.device:
    if str(device_arg).lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def save_six_panel(
    raw_base: torch.Tensor,
    raw_donor: torch.Tensor,
    raw_mix: torch.Tensor,
    detail_base: torch.Tensor,
    detail_donor: torch.Tensor,
    detail_mix: torch.Tensor,
    output_path: Path,
) -> None:
    panel = make_grid(
        [
            raw_base.detach().cpu(),
            raw_donor.detach().cpu(),
            raw_mix.detach().cpu(),
            detail_base.detach().cpu(),
            detail_donor.detach().cpu(),
            detail_mix.detach().cpu(),
        ],
        nrow=3,
        padding=4,
    )
    save_image(panel, str(output_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize dual-view cutmix_yolo using 6-panel images.")
    parser.add_argument("--raw-train-dir", type=str, default=DEFAULT_RAW_TRAIN_DIR)
    parser.add_argument("--detail-train-dir", type=str, default=DEFAULT_DETAIL_TRAIN_DIR)
    parser.add_argument("--cache-path", type=str, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cutmix-enable", type=int, default=DEFAULT_CUTMIX_ENABLE, choices=[0, 1])
    parser.add_argument("--cutmix-prob", type=float, default=DEFAULT_CUTMIX_PROB)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--n-show-per-batch", type=int, default=DEFAULT_N_SHOW_PER_BATCH)
    parser.add_argument("--enable-center-shift", type=int, default=DEFAULT_ENABLE_CENTER_SHIFT, choices=[0, 1])
    parser.add_argument("--bg-aug-enable", type=int, default=DEFAULT_BG_AUG_ENABLE, choices=[0, 1])
    parser.add_argument("--bg-aug-prob", type=float, default=DEFAULT_BG_AUG_PROB)
    parser.add_argument("--bg-bleed-into-box-ratio", type=float, default=DEFAULT_BG_AUG_BLEED_INTO_BOX_RATIO)
    parser.add_argument("--bg-black-dot-prob", type=float, default=DEFAULT_BG_AUG_BLACK_DOT_PROB)
    parser.add_argument("--bg-black-dot-size-min", type=int, default=DEFAULT_BG_AUG_BLACK_DOT_SIZE_MIN)
    parser.add_argument("--bg-black-dot-size-max", type=int, default=DEFAULT_BG_AUG_BLACK_DOT_SIZE_MAX)
    parser.add_argument("--bg-blur-sigma-min", type=float, default=DEFAULT_BG_AUG_BLUR_SIGMA_MIN)
    parser.add_argument("--bg-blur-sigma-max", type=float, default=DEFAULT_BG_AUG_BLUR_SIGMA_MAX)
    parser.add_argument("--bg-brightness", type=float, default=DEFAULT_BG_AUG_BRIGHTNESS)
    parser.add_argument("--bg-contrast", type=float, default=DEFAULT_BG_AUG_CONTRAST)
    parser.add_argument("--bg-saturation", type=float, default=DEFAULT_BG_AUG_SATURATION)
    parser.add_argument("--bg-hue", type=float, default=DEFAULT_BG_AUG_HUE)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.num_batches < 1:
        raise ValueError("--num-batches must be at least 1")
    if args.n_show_per_batch < 1:
        raise ValueError("--n-show-per-batch must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw_train_dir = resolve_existing_path(args.raw_train_dir)
    detail_train_dir = resolve_existing_path(args.detail_train_dir)
    cache_path = resolve_existing_path(args.cache_path)
    out_dir = resolve_output_path(args.out_dir)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    device = choose_runtime_device(args.device)
    print(f"raw_train_dir: {raw_train_dir}")
    print(f"detail_train_dir: {detail_train_dir}")
    print(f"cache_path: {cache_path}")
    print(f"out_dir: {out_dir}")
    print(f"device: {device}")

    items = build_paired_items(raw_train_dir, detail_train_dir)
    if len(items) < args.batch_size:
        raise RuntimeError(f"Not enough paired items: {len(items)} < batch_size={args.batch_size}")
    random.shuffle(items)

    helper_raw = build_helper(str(cache_path), str(raw_train_dir), enable_center_shift=bool(args.enable_center_shift))
    helper_detail = build_helper(str(cache_path), str(detail_train_dir), enable_center_shift=bool(args.enable_center_shift))

    rows: List[Dict[str, object]] = []
    summary = {
        "raw_train_dir": str(raw_train_dir),
        "detail_train_dir": str(detail_train_dir),
        "cache_path": str(cache_path),
        "out_dir": str(out_dir),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_batches": int(args.num_batches),
        "image_size": int(args.image_size),
        "seed": int(args.seed),
        "cutmix_enable": bool(args.cutmix_enable),
        "cutmix_prob": float(args.cutmix_prob),
        "alpha": float(args.alpha),
        "enable_center_shift": bool(args.enable_center_shift),
        "bg_aug_enable": bool(args.bg_aug_enable),
        "bg_aug_prob": float(args.bg_aug_prob),
        "bg_bleed_into_box_ratio": float(args.bg_bleed_into_box_ratio),
        "bg_black_dot_prob": float(args.bg_black_dot_prob),
        "bg_black_dot_size_min": int(args.bg_black_dot_size_min),
        "bg_black_dot_size_max": int(args.bg_black_dot_size_max),
        "bg_blur_sigma_min": float(args.bg_blur_sigma_min),
        "bg_blur_sigma_max": float(args.bg_blur_sigma_max),
        "bg_brightness": float(args.bg_brightness),
        "bg_contrast": float(args.bg_contrast),
        "bg_saturation": float(args.bg_saturation),
        "bg_hue": float(args.bg_hue),
        "num_batches_processed": 0,
        "samples_saved": 0,
    }

    ptr = 0
    for batch_idx in range(args.num_batches):
        batch_items = items[ptr : ptr + args.batch_size]
        if len(batch_items) < args.batch_size:
            break
        ptr += args.batch_size

        rel_paths = [item[0] for item in batch_items]
        raw_paths = [item[1] for item in batch_items]
        detail_paths = [item[2] for item in batch_items]
        labels = torch.tensor([item[3] for item in batch_items], dtype=torch.long, device=device)

        raw_images = torch.stack([load_image_tensor(path, args.image_size) for path in raw_paths], dim=0).to(device)
        detail_images = torch.stack([load_image_tensor(path, args.image_size) for path in detail_paths], dim=0).to(device)

        rng = random.Random(args.seed + batch_idx)
        apply_bg_aug = bool(args.bg_aug_enable) and (rng.random() < float(max(0.0, min(1.0, args.bg_aug_prob))))
        apply_cutmix = bool(args.cutmix_enable) and (rng.random() < float(max(0.0, min(1.0, args.cutmix_prob))))
        raw_base = []
        detail_base = []
        raw_shift_info = []
        detail_shift_info = []
        raw_bg_info = []
        detail_bg_info = []
        for i in range(args.batch_size):
            raw_shifted, raw_info = maybe_center_shift_image(raw_images[i], helper_raw, str(raw_paths[i]), rng)
            detail_shifted, detail_info = maybe_center_shift_image(detail_images[i], helper_detail, str(detail_paths[i]), rng)
            if apply_bg_aug:
                raw_shifted, raw_bg = apply_background_corruption(
                    raw_shifted,
                    raw_info.get("foreground_box"),
                    rng,
                    bleed_ratio=float(max(0.0, args.bg_bleed_into_box_ratio)),
                    black_dot_prob=float(max(0.0, args.bg_black_dot_prob)),
                    black_dot_size_min=int(max(1, args.bg_black_dot_size_min)),
                    black_dot_size_max=int(max(1, args.bg_black_dot_size_max)),
                    blur_sigma_min=float(max(0.0, args.bg_blur_sigma_min)),
                    blur_sigma_max=float(max(0.0, args.bg_blur_sigma_max)),
                    brightness=float(max(0.0, args.bg_brightness)),
                    contrast=float(max(0.0, args.bg_contrast)),
                    saturation=float(max(0.0, args.bg_saturation)),
                    hue=float(max(0.0, min(0.5, args.bg_hue))),
                )
                detail_shifted, detail_bg = apply_background_corruption(
                    detail_shifted,
                    detail_info.get("foreground_box"),
                    rng,
                    bleed_ratio=float(max(0.0, args.bg_bleed_into_box_ratio)),
                    black_dot_prob=float(max(0.0, args.bg_black_dot_prob)),
                    black_dot_size_min=int(max(1, args.bg_black_dot_size_min)),
                    black_dot_size_max=int(max(1, args.bg_black_dot_size_max)),
                    blur_sigma_min=float(max(0.0, args.bg_blur_sigma_min)),
                    blur_sigma_max=float(max(0.0, args.bg_blur_sigma_max)),
                    brightness=float(max(0.0, args.bg_brightness)),
                    contrast=float(max(0.0, args.bg_contrast)),
                    saturation=float(max(0.0, args.bg_saturation)),
                    hue=float(max(0.0, min(0.5, args.bg_hue))),
                )
            else:
                raw_bg = {"bg_aug_applied": False, "bg_mask_ratio": 0.0, "bg_preserve_box": None}
                detail_bg = {"bg_aug_applied": False, "bg_mask_ratio": 0.0, "bg_preserve_box": None}
            raw_base.append(raw_shifted)
            detail_base.append(detail_shifted)
            raw_shift_info.append(raw_info)
            detail_shift_info.append(detail_info)
            raw_bg_info.append(raw_bg)
            detail_bg_info.append(detail_bg)
        raw_base_t = torch.stack(raw_base, dim=0)
        detail_base_t = torch.stack(detail_base, dim=0)

        raw_mix_t = raw_base_t.clone()
        detail_mix_t = detail_base_t.clone()
        lam_batch = [1.0 for _ in range(args.batch_size)]
        theta_start_batch = [0.0 for _ in range(args.batch_size)]
        rand_idx = torch.arange(args.batch_size, device=device)
        span_angle = 0.0

        if apply_cutmix:
            raw_area_ratios = [
                helper_raw._estimate_area_ratio_from_valid_boxes(
                    lookup_valid_boxes(helper_raw, str(path), args.image_size, args.image_size),
                    args.image_size,
                    args.image_size,
                )
                for path in raw_paths
            ]
            rand_idx = helper_raw._build_pair_indices(raw_area_ratios, device=device, stats=PairStatsProxy())

            lam_scalar = float(np.random.beta(args.alpha, args.alpha)) if args.alpha > 0 else 1.0
            span_ratio = float(np.clip(1.0 - lam_scalar, 0.0, 1.0))
            span_angle = span_ratio * 2.0 * np.pi

            for i in range(args.batch_size):
                center_x, center_y = helper_raw._sample_sector_center(h=args.image_size, w=args.image_size)
                theta_start = float(np.random.uniform(-np.pi, np.pi))
                theta_end = theta_start + span_angle
                raw_mask = helper_raw._build_sector_mask(
                    h=args.image_size, w=args.image_size, center_x=center_x, center_y=center_y,
                    theta_start=theta_start, theta_end=theta_end, device=device,
                )
                detail_mask = helper_detail._build_sector_mask(
                    h=args.image_size, w=args.image_size, center_x=center_x, center_y=center_y,
                    theta_start=theta_start, theta_end=theta_end, device=device,
                )
                donor_idx = int(rand_idx[i].item())
                raw_mix_t[i] = torch.where(raw_mask.unsqueeze(0).expand_as(raw_mix_t[i]), raw_base_t[donor_idx], raw_base_t[i])
                detail_mix_t[i] = torch.where(detail_mask.unsqueeze(0).expand_as(detail_mix_t[i]), detail_base_t[donor_idx], detail_base_t[i])
                lam_batch[i] = 1.0 - float(raw_mask.float().mean().item())
                theta_start_batch[i] = theta_start

        save_count = min(args.n_show_per_batch, args.batch_size)
        for i in range(save_count):
            donor_idx = int(rand_idx[i].item())
            filename = (
                f"b{batch_idx:03d}_i{i:03d}_d{donor_idx:03d}"
                f"_lam{lam_batch[i]:.4f}_la{int(labels[i].item())}_lb{int(labels[donor_idx].item())}.png"
            )
            save_six_panel(
                raw_base=raw_base_t[i],
                raw_donor=raw_base_t[donor_idx],
                raw_mix=raw_mix_t[i],
                detail_base=detail_base_t[i],
                detail_donor=detail_base_t[donor_idx],
                detail_mix=detail_mix_t[i],
                output_path=panels_dir / filename,
            )
            rows.append(
                {
                    "batch": batch_idx,
                    "sample_index": i,
                    "donor_index": donor_idx,
                    "rel_path": rel_paths[i],
                    "label_a": int(labels[i].item()),
                    "label_b": int(labels[donor_idx].item()),
                    "lam": float(lam_batch[i]),
                    "theta_start": float(theta_start_batch[i]),
                    "span_angle": float(span_angle),
                    "raw_path": str(raw_paths[i]),
                    "detail_path": str(detail_paths[i]),
                    "raw_donor_path": str(raw_paths[donor_idx]),
                    "detail_donor_path": str(detail_paths[donor_idx]),
                     "enable_center_shift": bool(args.enable_center_shift),
                     "raw_shift_dx": raw_shift_info[i]["shift_dx"],
                     "raw_shift_dy": raw_shift_info[i]["shift_dy"],
                     "detail_shift_dx": detail_shift_info[i]["shift_dx"],
                     "detail_shift_dy": detail_shift_info[i]["shift_dy"],
                     "raw_foreground_box": raw_shift_info[i]["foreground_box"],
                     "detail_foreground_box": detail_shift_info[i]["foreground_box"],
                     "raw_bg_aug_applied": raw_bg_info[i]["bg_aug_applied"],
                     "detail_bg_aug_applied": detail_bg_info[i]["bg_aug_applied"],
                     "raw_bg_mask_ratio": raw_bg_info[i]["bg_mask_ratio"],
                     "detail_bg_mask_ratio": detail_bg_info[i]["bg_mask_ratio"],
                     "raw_bg_preserve_box": raw_bg_info[i]["bg_preserve_box"],
                     "detail_bg_preserve_box": detail_bg_info[i]["bg_preserve_box"],
                     "apply_bg_aug": bool(apply_bg_aug),
                     "apply_cutmix": bool(apply_cutmix),
                 }
             )

        summary["num_batches_processed"] += 1
        summary["samples_saved"] += save_count
        print(f"[Batch {batch_idx}] processed={args.batch_size} saved={save_count}")

    if summary["num_batches_processed"] == 0:
        raise RuntimeError("No batches were processed. Reduce batch size or check paired samples.")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if rows:
        with open(out_dir / "sample_report.csv", "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("Saved outputs:")
    print(panels_dir)
    print(out_dir / "summary.json")
    print(out_dir / "sample_report.csv")


if __name__ == "__main__":
    main()
