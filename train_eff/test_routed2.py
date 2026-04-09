import csv
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config4test2 import RoutedTrain2TestConfig
from model import build_train2_model
from utils import load_checkpoint


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VALID_POLICIES = {"raw_only", "crop224_only", "average"}


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


def build_rel_map(root: Path) -> Dict[str, Path]:
    items = {}
    for path in list_images(root):
        rel = to_unix(os.path.relpath(str(path), str(root)))
        items[rel] = path
    return items


def build_canonical_test_order(root: Path) -> List[str]:
    ordered_rel_paths: List[str] = []
    for img_name in os.listdir(root):
        img_path = root / img_name
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        ordered_rel_paths.append(to_unix(img_name))
    return ordered_rel_paths


def build_key(img_path: Path, root_dir: Path, key_mode: str) -> str:
    if key_mode == "relative_to_train_dir":
        key = os.path.relpath(str(img_path), str(root_dir))
    else:
        key = os.path.abspath(str(img_path))
    return to_unix(key)


def candidate_keys(img_path: Path, root_dir: Path, key_mode: str) -> List[str]:
    keys = [to_unix(str(img_path)), to_unix(os.path.normcase(os.path.abspath(str(img_path))))]

    if key_mode == "relative_to_train_dir":
        try:
            rel = os.path.relpath(
                os.path.normcase(os.path.abspath(str(img_path))),
                os.path.normcase(os.path.abspath(str(root_dir))),
            )
            keys.append(to_unix(rel))
        except Exception:
            pass

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


def lookup_entry(img_path: Path, root_dir: Path, key_mode: str, items: Dict[str, object]) -> Optional[object]:
    direct_key = build_key(img_path, root_dir, key_mode)
    if direct_key in items:
        return items[direct_key]

    for key in candidate_keys(img_path, root_dir, key_mode):
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


def clip_box_to_size(box: Sequence[float], width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    if len(box) < 4 or width <= 0 or height <= 0:
        return None

    x1 = max(0.0, min(float(width - 1), float(box[0])))
    y1 = max(0.0, min(float(height - 1), float(box[1])))
    x2 = max(x1 + 1.0, min(float(width), float(box[2])))
    y2 = max(y1 + 1.0, min(float(height), float(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def select_first_valid_box(boxes_raw: List[List[float]], orig_size: Optional[Tuple[int, int]]) -> Optional[Tuple[float, float, float, float]]:
    if orig_size is None:
        return None

    width, height = orig_size
    for box in boxes_raw:
        clipped = clip_box_to_size(box, width=width, height=height)
        if clipped is not None:
            return clipped
    return None


def compute_area_ratio(box_xyxy: Tuple[float, float, float, float], orig_size: Tuple[int, int]) -> Optional[float]:
    width, height = orig_size
    if width <= 0 or height <= 0:
        return None

    x1, y1, x2, y2 = box_xyxy
    box_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    image_area = float(width) * float(height)
    if image_area <= 0.0:
        return None
    return box_area / image_area


def validate_policy(policy: str, name: str) -> str:
    if policy not in VALID_POLICIES:
        raise ValueError(f"Unsupported {name}: {policy}. Expected one of {sorted(VALID_POLICIES)}")
    return policy


def policy_for_area_ratio(area_ratio: Optional[float], config) -> Tuple[str, str]:
    if area_ratio is None:
        return validate_policy(config.NO_BOX_POLICY, "NO_BOX_POLICY"), "no_box"
    if area_ratio < float(config.SMALL_AREA_THRES):
        return validate_policy(config.SMALL_POLICY, "SMALL_POLICY"), "small"
    if area_ratio < float(config.LARGE_AREA_THRES):
        return validate_policy(config.MID_POLICY, "MID_POLICY"), "mid"
    return validate_policy(config.LARGE_POLICY, "LARGE_POLICY"), "large"


class InferenceDataset(Dataset):
    def __init__(self, path_items: Sequence[Tuple[str, Path]], image_size: int):
        self.path_items = list(path_items)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.path_items)

    def __getitem__(self, idx: int):
        rel_path, img_path = self.path_items[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1)) / 255.0
        return image, rel_path


def collate_inference(batch):
    images, rel_paths = zip(*batch)
    return torch.stack(images, dim=0), list(rel_paths)


def normalize_rgb_batch(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def make_singleview_runtime_config(*, model_type: str, image_size: int, num_classes: int, pretrained_weights_dir: str, dropout: float, use_cbam_stage3: bool, cbam_reduction: int, cbam_spatial_kernel: int, single_view_source: str):
    runtime = SimpleNamespace()
    runtime.TRAIN_MODE = "single_view"
    runtime.SINGLE_VIEW_SOURCE = single_view_source
    runtime.MODEL_TYPE = model_type
    runtime.IMAGE_SIZE = int(image_size)
    runtime.NUM_CLASSES = int(num_classes)
    runtime.PRETRAINED_WEIGHTS_DIR = pretrained_weights_dir
    runtime.DROPOUT = float(dropout)
    runtime.USE_FREQ_CHANNELS = False
    runtime.USE_CBAM_STAGE3 = bool(use_cbam_stage3)
    runtime.CBAM_REDUCTION = int(cbam_reduction)
    runtime.CBAM_SPATIAL_KERNEL = int(cbam_spatial_kernel)
    return runtime


def load_model(model_config, checkpoint_path: Path, device: torch.device):
    model = build_train2_model(model_config, pretrained=False)
    model = model.to(device)
    load_checkpoint(model, None, str(checkpoint_path), device=device, verbose=True)
    model.eval()
    return model


def run_inference(
    model,
    path_items: Sequence[Tuple[str, Path]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    image_size: int,
    desc: str,
) -> Dict[str, np.ndarray]:
    if not path_items:
        return {}

    dataset = InferenceDataset(path_items, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_inference,
    )

    outputs: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for images, rel_paths in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            logits = model(normalize_rgb_batch(images))["fusion_logits"]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for rel_path, prob in zip(rel_paths, probs):
                outputs[rel_path] = prob
    return outputs


def save_results(results: Sequence[Tuple[str, int]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rel_path, pred in results:
            f.write(f"{Path(rel_path).name} {pred}\n")


def save_results_csv(rows: Sequence[Dict[str, object]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "predicted_label", "confidence", "route_bucket", "route_policy", "area_ratio"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    config = RoutedTrain2TestConfig()
    config_dir = Path(__file__).resolve().parent

    raw_test_dir = resolve_config_path(config.RAW_TEST_DIR, config_dir=config_dir, must_exist=True)
    detail_test_dir = resolve_config_path(config.DETAIL_TEST_DIR, config_dir=config_dir, must_exist=True)
    class_file = resolve_config_path(config.CLASSNAME_FILE, config_dir=config_dir, must_exist=True)
    cache_path = resolve_config_path(config.YOLO_TEST_CACHE_PATH, config_dir=config_dir, must_exist=True)
    raw_ckpt = resolve_config_path(config.RAW_CHECKPOINT_PATH, config_dir=config_dir, must_exist=True)
    detail_ckpt = resolve_config_path(config.DETAIL_CHECKPOINT_PATH, config_dir=config_dir, must_exist=True)
    output_file = resolve_config_path(config.OUTPUT_FILE, config_dir=config_dir, must_exist=False)
    output_csv = resolve_config_path(config.OUTPUT_CSV, config_dir=config_dir, must_exist=False) if getattr(config, "OUTPUT_CSV", None) else None

    if not raw_test_dir.exists():
        raise FileNotFoundError(f"RAW_TEST_DIR not found: {raw_test_dir}")
    if not detail_test_dir.exists():
        raise FileNotFoundError(f"DETAIL_TEST_DIR not found: {detail_test_dir}")
    if not class_file.exists():
        raise FileNotFoundError(f"CLASSNAME_FILE not found: {class_file}")
    if not cache_path.exists():
        raise FileNotFoundError(f"YOLO_TEST_CACHE_PATH not found: {cache_path}")
    if not raw_ckpt.exists():
        raise FileNotFoundError(f"RAW_CHECKPOINT_PATH not found: {raw_ckpt}")
    if not detail_ckpt.exists():
        raise FileNotFoundError(f"DETAIL_CHECKPOINT_PATH not found: {detail_ckpt}")

    raw_map = build_rel_map(raw_test_dir)
    detail_map = build_rel_map(detail_test_dir)
    if not raw_map:
        raise RuntimeError(f"No test images found under: {raw_test_dir}")

    canonical_rel_paths = build_canonical_test_order(raw_test_dir)
    if not canonical_rel_paths:
        raise RuntimeError(f"No test images found in canonical order under: {raw_test_dir}")

    raw_rel_paths = set(raw_map.keys())
    detail_rel_paths = set(detail_map.keys())
    missing_in_detail = sorted(raw_rel_paths - detail_rel_paths)
    missing_in_raw = sorted(detail_rel_paths - raw_rel_paths)
    if missing_in_detail:
        raise RuntimeError(
            f"Detail test set is missing {len(missing_in_detail)} files. First missing: {missing_in_detail[0]}"
        )
    if missing_in_raw:
        raise RuntimeError(
            f"Raw test set is missing {len(missing_in_raw)} files that exist in detail set. First extra: {missing_in_raw[0]}"
        )

    missing_in_raw_map = [rel_path for rel_path in canonical_rel_paths if rel_path not in raw_map]
    if missing_in_raw_map:
        raise RuntimeError(
            f"Canonical test order includes files missing from raw map. First missing: {missing_in_raw_map[0]}"
        )

    with open(class_file, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    _, cache_items = load_cache(cache_path)

    route_info: Dict[str, Dict[str, object]] = {}
    stats = Counter()
    raw_needed = set()
    detail_needed = set()

    for rel_path in canonical_rel_paths:
        raw_path = raw_map[rel_path]
        entry = lookup_entry(raw_path, raw_test_dir, config.YOLO_KEY_MODE, cache_items)
        boxes_raw, orig_size = parse_entry(entry)

        if orig_size is None:
            with Image.open(raw_path) as image:
                orig_size = image.size

        chosen_box = select_first_valid_box(boxes_raw, orig_size) if boxes_raw else None
        area_ratio = compute_area_ratio(chosen_box, orig_size) if (chosen_box and orig_size) else None
        policy, bucket = policy_for_area_ratio(area_ratio, config)

        route_info[rel_path] = {
            "bucket": bucket,
            "policy": policy,
            "area_ratio": area_ratio,
        }
        stats[f"{bucket}_count"] += 1
        stats[f"{policy}_count"] += 1

        if policy in {"raw_only", "average"}:
            raw_needed.add(rel_path)
        if policy in {"crop224_only", "average"}:
            detail_needed.add(rel_path)

    if int(config.NUM_CLASSES) != len(class_names):
        raise ValueError(f"CLASSNAME_FILE has {len(class_names)} classes but config expects {config.NUM_CLASSES}")

    raw_runtime_config = make_singleview_runtime_config(
        model_type=config.RAW_MODEL_TYPE,
        image_size=config.RAW_IMAGE_SIZE,
        num_classes=config.NUM_CLASSES,
        pretrained_weights_dir=config.PRETRAINED_WEIGHTS_DIR,
        dropout=config.DROPOUT,
        use_cbam_stage3=config.USE_CBAM_STAGE3,
        cbam_reduction=config.CBAM_REDUCTION,
        cbam_spatial_kernel=config.CBAM_SPATIAL_KERNEL,
        single_view_source="raw",
    )
    detail_runtime_config = make_singleview_runtime_config(
        model_type=config.DETAIL_MODEL_TYPE,
        image_size=config.DETAIL_IMAGE_SIZE,
        num_classes=config.NUM_CLASSES,
        pretrained_weights_dir=config.PRETRAINED_WEIGHTS_DIR,
        dropout=config.DROPOUT,
        use_cbam_stage3=config.USE_CBAM_STAGE3,
        cbam_reduction=config.CBAM_REDUCTION,
        cbam_spatial_kernel=config.CBAM_SPATIAL_KERNEL,
        single_view_source="detail",
    )

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Raw test dir: {raw_test_dir}")
    print(f"Detail test dir: {detail_test_dir}")
    print(f"YOLO cache: {cache_path}")
    print(f"Raw checkpoint: {raw_ckpt}")
    print(f"Detail checkpoint: {detail_ckpt}")
    print(
        "Routing policies: "
        f"small={config.SMALL_POLICY}, mid={config.MID_POLICY}, "
        f"large={config.LARGE_POLICY}, no_box={config.NO_BOX_POLICY}"
    )

    print("\nLoading raw single-view model...")
    raw_model = load_model(raw_runtime_config, raw_ckpt, device)
    print("\nLoading detail single-view model...")
    detail_model = load_model(detail_runtime_config, detail_ckpt, device)

    raw_items = [(rel_path, raw_map[rel_path]) for rel_path in canonical_rel_paths if rel_path in raw_needed]
    detail_items = [(rel_path, detail_map[rel_path]) for rel_path in canonical_rel_paths if rel_path in detail_needed]

    raw_outputs = run_inference(
        raw_model,
        raw_items,
        device=device,
        batch_size=int(config.BATCH_SIZE),
        num_workers=int(config.NUM_WORKERS),
        pin_memory=bool(config.PIN_MEMORY),
        image_size=int(config.RAW_IMAGE_SIZE),
        desc="Raw single-view inference",
    )
    detail_outputs = run_inference(
        detail_model,
        detail_items,
        device=device,
        batch_size=int(config.BATCH_SIZE),
        num_workers=int(config.NUM_WORKERS),
        pin_memory=bool(config.PIN_MEMORY),
        image_size=int(config.DETAIL_IMAGE_SIZE),
        desc="Detail single-view inference",
    )

    results: List[Tuple[str, int]] = []
    csv_rows: List[Dict[str, object]] = []
    for rel_path in canonical_rel_paths:
        policy = route_info[rel_path]["policy"]
        area_ratio = route_info[rel_path]["area_ratio"]
        bucket = route_info[rel_path]["bucket"]

        if policy == "raw_only":
            probs = raw_outputs[rel_path]
        elif policy == "crop224_only":
            probs = detail_outputs[rel_path]
        elif policy == "average":
            probs = 0.5 * (raw_outputs[rel_path] + detail_outputs[rel_path])
        else:
            raise RuntimeError(f"Unexpected policy for {rel_path}: {policy}")

        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        results.append((rel_path, pred))
        csv_rows.append(
            {
                "image": Path(rel_path).name,
                "predicted_label": pred,
                "confidence": conf,
                "route_bucket": bucket,
                "route_policy": policy,
                "area_ratio": "" if area_ratio is None else float(area_ratio),
            }
        )

    save_results(results, output_file)
    print(f"\nResults saved to: {output_file}")
    if output_csv is not None:
        save_results_csv(csv_rows, output_csv)
        print(f"Detailed results saved to: {output_csv}")

    print("\nRouting statistics:")
    for key in [
        "small_count",
        "mid_count",
        "large_count",
        "no_box_count",
        "raw_only_count",
        "crop224_only_count",
        "average_count",
    ]:
        print(f"  {key}: {stats.get(key, 0)}")

    print("\nFirst 10 predictions:")
    for rel_path, pred in results[:10]:
        print(f"  {Path(rel_path).name} {pred}")


if __name__ == "__main__":
    main()
