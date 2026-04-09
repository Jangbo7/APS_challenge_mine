import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

from config4yolo import YoloPrecomputeConfig


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def to_unix(path: str) -> str:
    return path.replace("\\", "/").strip()


def resolve_config_path(path_value: str, *, config_dir: Path, must_exist: bool) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    config_candidate = (config_dir / path).resolve()
    candidates = [cwd_candidate]
    if config_candidate != cwd_candidate:
        candidates.append(config_candidate)

    if must_exist:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]

    return config_candidate


def list_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                files.append(Path(dp) / fn)
    files.sort()
    return files


def build_key(img_path: Path, train_dir: Path, key_mode: str) -> str:
    if key_mode == "relative_to_train_dir":
        key = os.path.relpath(str(img_path), str(train_dir))
    else:
        key = os.path.abspath(str(img_path))
    return to_unix(key)


def build_force_resize_lookup(samples: Sequence[str]) -> set[str]:
    lookup = set()
    for sample in samples:
        if not isinstance(sample, str):
            continue
        sample_norm = to_unix(sample)
        if sample_norm:
            lookup.add(sample_norm)
    return lookup


def should_force_resize(img_path: Path, rel_path: Path, lookup: set[str]) -> bool:
    if not lookup:
        return False
    return img_path.name in lookup or to_unix(str(rel_path)) in lookup


def candidate_keys(img_path: Path, train_dir: Path, key_mode: str) -> List[str]:
    keys = []
    p_unix = to_unix(str(img_path))
    keys.append(p_unix)

    img_abs = os.path.normcase(os.path.abspath(str(img_path)))
    img_abs_unix = to_unix(img_abs)
    keys.append(img_abs_unix)

    if key_mode == "relative_to_train_dir":
        try:
            rel = os.path.relpath(img_abs, os.path.normcase(os.path.abspath(str(train_dir))))
            keys.append(to_unix(rel))
        except Exception:
            pass

    marker = "/train/"
    if marker in p_unix:
        keys.append(p_unix.split(marker, 1)[1])
    if marker in img_abs_unix:
        keys.append(img_abs_unix.split(marker, 1)[1])

    seen = set()
    out = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def load_cache(cache_path: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "items" in data:
        meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}
        raw_items = data.get("items", {})
    else:
        meta = {}
        raw_items = data

    if not isinstance(raw_items, dict):
        raise ValueError(f"Invalid YOLO cache format: items should be dict, got {type(raw_items)}")

    items = {to_unix(str(k)): v for k, v in raw_items.items()}
    return meta, items


def lookup_entry(img_path: Path, train_dir: Path, key_mode: str, items: Dict[str, object]) -> Optional[object]:
    direct_key = build_key(img_path, train_dir, key_mode)
    if direct_key in items:
        return items[direct_key]

    for key in candidate_keys(img_path, train_dir, key_mode):
        if key in items:
            return items[key]
    return None


def parse_entry(entry: object) -> Tuple[List[List[float]], Optional[Tuple[int, int]]]:
    if entry is None:
        return [], None
    if isinstance(entry, list):
        return entry, None
    if isinstance(entry, dict):
        boxes = entry.get("boxes", [])
        orig_size = entry.get("orig_size")
        if (
            isinstance(orig_size, (list, tuple))
            and len(orig_size) >= 2
            and float(orig_size[0]) > 0
            and float(orig_size[1]) > 0
        ):
            return boxes if isinstance(boxes, list) else [], (int(orig_size[0]), int(orig_size[1]))
        return boxes if isinstance(boxes, list) else [], None
    return [], None


def map_box_to_current_size(
    box: Sequence[float],
    orig_size: Optional[Tuple[int, int]],
    width: int,
    height: int,
) -> Sequence[float]:
    if orig_size is None:
        return box

    ow, oh = orig_size
    if ow <= 0 or oh <= 0:
        return box

    sx = float(width) / float(ow)
    sy = float(height) / float(oh)
    mapped = list(box)
    if len(mapped) >= 4:
        mapped[0] = float(mapped[0]) * sx
        mapped[1] = float(mapped[1]) * sy
        mapped[2] = float(mapped[2]) * sx
        mapped[3] = float(mapped[3]) * sy
    return mapped


def clip_box_to_image(box: Sequence[float], width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    if len(box) < 4:
        return None

    x1 = max(0.0, min(float(width - 1), float(box[0])))
    y1 = max(0.0, min(float(height - 1), float(box[1])))
    x2 = max(x1 + 1.0, min(float(width), float(box[2])))
    y2 = max(y1 + 1.0, min(float(height), float(box[3])))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def select_box(
    boxes_raw: List[List[float]],
    orig_size: Optional[Tuple[int, int]],
    width: int,
    height: int,
    box_select: str,
) -> Optional[Tuple[float, float, float, float]]:
    if box_select != "first_valid":
        raise ValueError(f"Unsupported PREPROCESS_BOX_SELECT: {box_select}")

    for box in boxes_raw:
        mapped = map_box_to_current_size(box, orig_size, width=width, height=height)
        clipped = clip_box_to_image(mapped, width=width, height=height)
        if clipped is not None:
            return clipped
    return None


def shifted_square_crop(
    center_x: float,
    center_y: float,
    side_length: float,
    width: int,
    height: int,
) -> Tuple[Tuple[int, int, int, int], bool, bool]:
    effective_side = max(1.0, min(float(side_length), float(width), float(height)))
    reduced = effective_side + 1e-6 < float(side_length)

    half = 0.5 * effective_side
    x1 = center_x - half
    y1 = center_y - half

    max_x1 = max(0.0, float(width) - effective_side)
    max_y1 = max(0.0, float(height) - effective_side)
    new_x1 = min(max(x1, 0.0), max_x1)
    new_y1 = min(max(y1, 0.0), max_y1)
    shifted = abs(new_x1 - x1) > 1e-6 or abs(new_y1 - y1) > 1e-6

    x1_i = int(round(new_x1))
    y1_i = int(round(new_y1))
    side_i = max(1, min(int(round(effective_side)), width, height))

    if x1_i + side_i > width:
        x1_i = max(0, width - side_i)
    if y1_i + side_i > height:
        y1_i = max(0, height - side_i)

    x2_i = x1_i + side_i
    y2_i = y1_i + side_i
    return (x1_i, y1_i, x2_i, y2_i), shifted, reduced


def center_crop_box(width: int, height: int, side_length: float) -> Tuple[Tuple[int, int, int, int], bool, bool]:
    return shifted_square_crop(0.5 * float(width), 0.5 * float(height), side_length, width, height)


def detection_crop_box(
    box_xyxy: Tuple[float, float, float, float],
    side_length: float,
    width: int,
    height: int,
) -> Tuple[Tuple[int, int, int, int], bool, bool]:
    x1, y1, x2, y2 = box_xyxy
    return shifted_square_crop(0.5 * (x1 + x2), 0.5 * (y1 + y2), side_length, width, height)


def save_resized_crop(image: Image.Image, crop_box: Tuple[int, int, int, int], output_path: Path, target_size: int) -> None:
    cropped = image.crop(crop_box)
    if cropped.size != (target_size, target_size):
        cropped = cropped.resize((target_size, target_size), Image.Resampling.BILINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output_path)


def save_resized_full_image(image: Image.Image, output_path: Path, target_size: int) -> None:
    resized = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path)


def main() -> None:
    config = YoloPrecomputeConfig()
    config_dir = Path(__file__).resolve().parent

    input_dir = resolve_config_path(config.PREPROCESS_INPUT_DIR, config_dir=config_dir, must_exist=True)
    output_dir = resolve_config_path(config.PREPROCESS_OUTPUT_DIR, config_dir=config_dir, must_exist=False)
    cache_path = resolve_config_path(config.PREPROCESS_CACHE_PATH, config_dir=config_dir, must_exist=True)
    key_mode = config.KEY_MODE

    if not input_dir.exists():
        raise FileNotFoundError(f"Preprocess input dir not found: {input_dir}")
    if not cache_path.exists():
        raise FileNotFoundError(f"Preprocess cache not found: {cache_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not getattr(config, "PREPROCESS_OVERWRITE", False):
        raise FileExistsError(f"Output dir is not empty: {output_dir}. Set PREPROCESS_OVERWRITE=True to reuse it.")

    target_size = int(getattr(config, "PREPROCESS_TARGET_SIZE", 224))
    if target_size <= 0:
        raise ValueError("PREPROCESS_TARGET_SIZE must be positive")

    no_box_policy = getattr(config, "PREPROCESS_NO_BOX_POLICY", "center_crop_224")
    if no_box_policy != "center_crop_224":
        raise ValueError(f"Unsupported PREPROCESS_NO_BOX_POLICY: {no_box_policy}")

    oob_policy = getattr(config, "PREPROCESS_OUT_OF_BOUNDS_POLICY", "shift_into_image")
    if oob_policy != "shift_into_image":
        raise ValueError(f"Unsupported PREPROCESS_OUT_OF_BOUNDS_POLICY: {oob_policy}")

    box_select = getattr(config, "PREPROCESS_BOX_SELECT", "first_valid")
    small_area_thres = float(getattr(config, "PREPROCESS_SMALL_AREA_THRES", 0.05))
    large_area_thres = float(getattr(config, "PREPROCESS_LARGE_AREA_THRES", 0.307))
    linear_a = float(getattr(config, "PREPROCESS_LINEAR_A", 1120.0))
    linear_b = float(getattr(config, "PREPROCESS_LINEAR_B", 168.0))

    if large_area_thres < small_area_thres:
        raise ValueError("PREPROCESS_LARGE_AREA_THRES must be >= PREPROCESS_SMALL_AREA_THRES")

    _, items = load_cache(cache_path)
    images = list_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found under: {input_dir}")

    stats = {
        "total": 0,
        "small_box_crop": 0,
        "mid_box_crop": 0,
        "large_box_resize": 0,
        "forced_resize_override": 0,
        "no_box_center_crop": 0,
        "box_lookup_miss": 0,
        "invalid_box_entry": 0,
        "shifted_into_image": 0,
        "reduced_side_for_small_image": 0,
    }
    force_resize_lookup = build_force_resize_lookup(
        getattr(config, "PREPROCESS_FORCE_RESIZE_SAMPLES", [])
    )

    pbar = tqdm(images, desc="YOLO crop 224 preprocess")
    for img_path in pbar:
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        stats["total"] += 1

        with Image.open(img_path) as image_in:
            image = image_in.convert("RGB")
            width, height = image.size

            if should_force_resize(img_path, rel_path, force_resize_lookup):
                stats["forced_resize_override"] += 1
                save_resized_full_image(image, out_path, target_size)
                continue

            entry = lookup_entry(img_path, input_dir, key_mode, items)
            boxes_raw, orig_size = parse_entry(entry)
            chosen_box = select_box(boxes_raw, orig_size, width, height, box_select) if boxes_raw else None

            if entry is None:
                stats["box_lookup_miss"] += 1
            elif chosen_box is None:
                stats["invalid_box_entry"] += 1

            if chosen_box is None:
                crop_box, shifted, reduced = center_crop_box(width, height, target_size)
                if shifted:
                    stats["shifted_into_image"] += 1
                if reduced:
                    stats["reduced_side_for_small_image"] += 1
                stats["no_box_center_crop"] += 1
                save_resized_crop(image, crop_box, out_path, target_size)
                continue

            x1, y1, x2, y2 = chosen_box
            box_area = max(1.0, (x2 - x1) * (y2 - y1))
            image_area = max(1.0, float(width * height))
            area_ratio = box_area / image_area

            if area_ratio < small_area_thres:
                crop_box, shifted, reduced = detection_crop_box(chosen_box, target_size, width, height)
                if shifted:
                    stats["shifted_into_image"] += 1
                if reduced:
                    stats["reduced_side_for_small_image"] += 1
                stats["small_box_crop"] += 1
                save_resized_crop(image, crop_box, out_path, target_size)
            elif area_ratio < large_area_thres:
                crop_side = linear_a * area_ratio + linear_b
                crop_box, shifted, reduced = detection_crop_box(chosen_box, crop_side, width, height)
                if shifted:
                    stats["shifted_into_image"] += 1
                if reduced:
                    stats["reduced_side_for_small_image"] += 1
                stats["mid_box_crop"] += 1
                save_resized_crop(image, crop_box, out_path, target_size)
            else:
                stats["large_box_resize"] += 1
                save_resized_full_image(image, out_path, target_size)

    print("\nDone.")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Cache path: {cache_path}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
