import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_TRAIN_DIR = Path("data/APS_dataset/train")
DEFAULT_TEST_IMAGE_DIR = Path("data/APS_dataset/test/test_noclass")
DEFAULT_TRAIN_CACHE = Path("eff/yolo_boxes_cache.json")
DEFAULT_TEST_CACHE = Path("eff/yolo_boxes_cache_test.json")
DEFAULT_CLASS_FILE = Path("data/APS_dataset/classname.txt")
DEFAULT_RESULT_CANDIDATES = [
    # Path("eff/eff2/result2.txt"),
    # Path("eff/result_routed.txt"),
    # Path("eff/result.txt"),
]
MIN_VALID_CLASS_SAMPLES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze test predictions with YOLO box area distributions from the training set "
            "and export test samples whose predicted-class area is outside the expected range."
        )
    )
    parser.add_argument(
        "--result-file",
        default=None,
        help="Path to a result txt file with 'filename pred' per line. If omitted, the script auto-detects one.",
    )
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR), help="Training image directory with class subfolders.")
    parser.add_argument("--test-image-dir", default=str(DEFAULT_TEST_IMAGE_DIR), help="Original test image directory.")
    parser.add_argument("--train-cache-path", default=str(DEFAULT_TRAIN_CACHE), help="YOLO cache path for the training set.")
    parser.add_argument("--test-cache-path", default=str(DEFAULT_TEST_CACHE), help="YOLO cache path for the test set.")
    parser.add_argument("--output-dir", default='eff/size_var', help="Directory to store the analysis outputs.")
    parser.add_argument("--class-file", default=str(DEFAULT_CLASS_FILE), help="Path to classname.txt.")
    parser.add_argument("--lower-quantile", type=float, default=0.05, help="Lower quantile used for outlier detection.")
    parser.add_argument("--upper-quantile", type=float, default=0.95, help="Upper quantile used for outlier detection.")
    return parser.parse_args()


def resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates = [
        (Path.cwd() / path).resolve(),
        (Path(__file__).resolve().parent / path).resolve(),
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


def find_default_result_file() -> Optional[Path]:
    for candidate in DEFAULT_RESULT_CANDIDATES:
        resolved = resolve_existing_path(str(candidate))
        if resolved.exists() and resolved.is_file():
            return resolved

    search_roots = [
        resolve_existing_path("eff"),
        resolve_existing_path("train_eff/eff"),
    ]
    seen = set()
    discovered: List[Path] = []
    for root in search_roots:
        root_key = str(root)
        if root_key in seen or not root.exists() or not root.is_dir():
            continue
        seen.add(root_key)
        for path in root.rglob("result*.txt"):
            if path.is_file():
                discovered.append(path.resolve())

    if not discovered:
        return None

    discovered = sorted(set(discovered), key=lambda p: p.stat().st_mtime, reverse=True)
    return discovered[0]


def to_unix_rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def parse_result_file(result_path: Path) -> Dict[str, int]:
    predictions: Dict[str, int] = {}
    with open(result_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid format in {result_path} line {line_no}: expected 'filename label', got {raw_line.rstrip()!r}"
                )
            filename, pred_str = parts
            filename = Path(filename).name
            try:
                pred = int(pred_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label in {result_path} line {line_no}: {pred_str!r} is not an integer"
                ) from exc
            if filename in predictions:
                raise ValueError(f"Duplicate filename in result file: {filename}")
            predictions[filename] = pred
    return predictions


def load_class_names(class_file: Path) -> List[str]:
    with open(class_file, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    if not class_names:
        raise RuntimeError(f"No class names found in: {class_file}")
    return class_names


def load_yolo_cache(cache_path: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    with open(cache_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected YOLO cache format in {cache_path}: top-level object must be a dict")

    if "items" in payload:
        items = payload.get("items")
        meta = payload.get("meta", {})
        if not isinstance(items, dict):
            raise ValueError(f"Unexpected YOLO cache format in {cache_path}: 'items' must be a dict")
        if not isinstance(meta, dict):
            meta = {}
        return items, meta

    items = {k: v for k, v in payload.items() if k != "meta"}
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    return items, meta


def parse_cache_entry(entry: object) -> Tuple[List[List[float]], Optional[Tuple[int, int]]]:
    if entry is None:
        return [], None

    if isinstance(entry, list):
        return entry, None

    if isinstance(entry, dict):
        boxes = entry.get("boxes", [])
        orig_size = entry.get("orig_size")
        parsed_size = None
        if (
            isinstance(orig_size, (list, tuple))
            and len(orig_size) >= 2
            and float(orig_size[0]) > 0
            and float(orig_size[1]) > 0
        ):
            parsed_size = (int(orig_size[0]), int(orig_size[1]))
        return boxes if isinstance(boxes, list) else [], parsed_size

    return [], None


def estimate_area_ratio_from_entry(entry: object) -> Optional[float]:
    boxes_raw, orig_size = parse_cache_entry(entry)
    if not isinstance(boxes_raw, list) or not boxes_raw:
        return None

    if orig_size is None:
        return None

    width, height = orig_size
    if width <= 0 or height <= 0:
        return None
    image_area = float(width) * float(height)
    if image_area <= 0:
        return None

    area_ratios: List[float] = []
    for box in boxes_raw:
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x1 = max(0.0, min(float(box[0]), float(width)))
        y1 = max(0.0, min(float(box[1]), float(height)))
        x2 = max(x1, min(float(box[2]), float(width)))
        y2 = max(y1, min(float(box[3]), float(height)))
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if box_area <= 0.0:
            continue
        area_ratios.append(box_area / image_area)

    if not area_ratios:
        return None
    return float(np.median(area_ratios))


def iter_images(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Image directory is not a directory: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def build_test_image_map(test_image_dir: Path) -> Dict[str, Path]:
    image_map: Dict[str, Path] = {}
    for image_path in iter_images(test_image_dir):
        name = image_path.name
        if name in image_map:
            raise ValueError(f"Duplicate test image filename found: {name}")
        image_map[name] = image_path
    if not image_map:
        raise RuntimeError(f"No test images found under: {test_image_dir}")
    return image_map


def build_test_cache_index(test_items: Dict[str, object]) -> Dict[str, object]:
    index: Dict[str, object] = {}
    for key, entry in test_items.items():
        name = Path(key).name
        if name in index:
            raise ValueError(f"Duplicate filename in test YOLO cache after basename normalization: {name}")
        index[name] = entry
    return index


def compute_class_area_stats(
    train_dir: Path,
    train_items: Dict[str, object],
    class_names: Sequence[str],
    lower_quantile: float,
    upper_quantile: float,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    class_name_set = set(class_names)
    grouped: Dict[str, List[float]] = defaultdict(list)
    class_total_counts: Counter = Counter()
    class_missing_cache_counts: Counter = Counter()
    class_invalid_area_counts: Counter = Counter()

    for image_path in iter_images(train_dir):
        rel = to_unix_rel(image_path, train_dir)
        class_name = Path(rel).parts[0]
        if class_name not in class_name_set:
            continue

        class_total_counts[class_name] += 1
        entry = train_items.get(rel)
        if entry is None:
            class_missing_cache_counts[class_name] += 1
            continue

        area_ratio = estimate_area_ratio_from_entry(entry)
        if area_ratio is None:
            class_invalid_area_counts[class_name] += 1
            continue

        grouped[class_name].append(area_ratio)

    stats_rows: List[Dict[str, object]] = []
    stats_by_class: Dict[str, Dict[str, object]] = {}
    for class_name in class_names:
        values = sorted(grouped.get(class_name, []))
        valid_count = len(values)
        total_count = int(class_total_counts.get(class_name, 0))
        too_few = valid_count < MIN_VALID_CLASS_SAMPLES

        row: Dict[str, object] = {
            "class_name": class_name,
            "train_total_images": total_count,
            "valid_area_count": valid_count,
            "missing_cache_count": int(class_missing_cache_counts.get(class_name, 0)),
            "invalid_area_count": int(class_invalid_area_counts.get(class_name, 0)),
            "too_few_samples": bool(too_few),
            "min_area": None,
            "max_area": None,
            "mean_area": None,
            "median_area": None,
            "p5": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "lower_bound": None,
            "upper_bound": None,
        }
        if values:
            arr = np.asarray(values, dtype=np.float64)
            row.update(
                {
                    "min_area": float(arr.min()),
                    "max_area": float(arr.max()),
                    "mean_area": float(arr.mean()),
                    "median_area": float(np.median(arr)),
                    "p5": float(np.quantile(arr, 0.05)),
                    "p25": float(np.quantile(arr, 0.25)),
                    "p50": float(np.quantile(arr, 0.50)),
                    "p75": float(np.quantile(arr, 0.75)),
                    "p95": float(np.quantile(arr, 0.95)),
                    "lower_bound": float(np.quantile(arr, lower_quantile)),
                    "upper_bound": float(np.quantile(arr, upper_quantile)),
                }
            )
        stats_rows.append(row)
        stats_by_class[class_name] = row

    return stats_rows, stats_by_class


def make_export_name(filename: str, pred_class: str, area_ratio: float, lower_bound: float, upper_bound: float) -> str:
    source = Path(filename)
    safe_class = pred_class.replace("/", "_").replace("\\", "_").replace(" ", "_")
    suffix = source.suffix if source.suffix else ".jpg"
    stem = source.stem if source.stem else source.name
    return (
        f"{stem}__pred[{safe_class}]__area[{area_ratio:.4f}]__range[{lower_bound:.4f}-{upper_bound:.4f}]"
        f"{suffix}"
    )


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def analyze_results(
    predictions: Dict[str, int],
    class_names: Sequence[str],
    class_stats: Dict[str, Dict[str, object]],
    test_image_map: Dict[str, Path],
    test_cache_index: Dict[str, object],
    outlier_dir: Path,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    outliers: List[Dict[str, object]] = []
    undetermined: List[Dict[str, object]] = []
    per_class_outliers: Counter = Counter()
    undetermined_reasons: Counter = Counter()

    for filename, pred_label in sorted(predictions.items()):
        if pred_label < 0 or pred_label >= len(class_names):
            undetermined_reasons["invalid_pred_label"] += 1
            undetermined.append(
                {
                    "image": filename,
                    "pred_label": pred_label,
                    "pred_class": None,
                    "test_area_ratio": None,
                    "source_image_path": None,
                    "reason": "invalid_pred_label",
                }
            )
            continue

        pred_class = class_names[pred_label]
        class_stat = class_stats.get(pred_class)
        if class_stat is None:
            undetermined_reasons["missing_class_stats"] += 1
            undetermined.append(
                {
                    "image": filename,
                    "pred_label": pred_label,
                    "pred_class": pred_class,
                    "test_area_ratio": None,
                    "source_image_path": None,
                    "reason": "missing_class_stats",
                }
            )
            continue

        if bool(class_stat.get("too_few_samples", False)):
            undetermined_reasons["too_few_train_samples"] += 1
            undetermined.append(
                {
                    "image": filename,
                    "pred_label": pred_label,
                    "pred_class": pred_class,
                    "test_area_ratio": None,
                    "source_image_path": None,
                    "reason": "too_few_train_samples",
                }
            )
            continue

        image_path = test_image_map.get(filename)
        if image_path is None:
            undetermined_reasons["test_image_not_found"] += 1
            undetermined.append(
                {
                    "image": filename,
                    "pred_label": pred_label,
                    "pred_class": pred_class,
                    "test_area_ratio": None,
                    "source_image_path": None,
                    "reason": "test_image_not_found",
                }
            )
            continue

        entry = test_cache_index.get(filename)
        if entry is None:
            undetermined_reasons["test_cache_missing"] += 1
            undetermined.append(
                {
                    "image": filename,
                    "pred_label": pred_label,
                    "pred_class": pred_class,
                    "test_area_ratio": None,
                    "source_image_path": str(image_path),
                    "reason": "test_cache_missing",
                }
            )
            continue

        area_ratio = estimate_area_ratio_from_entry(entry)
        if area_ratio is None:
            undetermined_reasons["test_area_invalid"] += 1
            undetermined.append(
                {
                    "image": filename,
                    "pred_label": pred_label,
                    "pred_class": pred_class,
                    "test_area_ratio": None,
                    "source_image_path": str(image_path),
                    "reason": "test_area_invalid",
                }
            )
            continue

        lower_bound = class_stat["lower_bound"]
        upper_bound = class_stat["upper_bound"]
        if lower_bound is None or upper_bound is None:
            undetermined_reasons["missing_bounds"] += 1
            undetermined.append(
                {
                    "image": filename,
                    "pred_label": pred_label,
                    "pred_class": pred_class,
                    "test_area_ratio": float(area_ratio),
                    "source_image_path": str(image_path),
                    "reason": "missing_bounds",
                }
            )
            continue

        deviation_side = None
        if area_ratio < float(lower_bound):
            deviation_side = "below"
        elif area_ratio > float(upper_bound):
            deviation_side = "above"

        if deviation_side is None:
            continue

        export_name = make_export_name(filename, pred_class, area_ratio, float(lower_bound), float(upper_bound))
        copied_path = outlier_dir / export_name
        shutil.copy2(image_path, copied_path)

        row = {
            "image": filename,
            "pred_label": pred_label,
            "pred_class": pred_class,
            "test_area_ratio": float(area_ratio),
            "train_p5": class_stat["p5"],
            "train_p95": class_stat["p95"],
            "train_median": class_stat["median_area"],
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "deviation_side": deviation_side,
            "source_image_path": str(image_path),
            "copied_path": str(copied_path),
        }
        outliers.append(row)
        per_class_outliers[pred_class] += 1

    summary = {
        "total_predictions": len(predictions),
        "determinable_count": len(predictions) - len(undetermined),
        "outlier_count": len(outliers),
        "undetermined_count": len(undetermined),
        "per_class_outlier_counts": dict(sorted(per_class_outliers.items())),
        "undetermined_reason_counts": dict(sorted(undetermined_reasons.items())),
    }
    return outliers, undetermined, summary


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.lower_quantile < args.upper_quantile <= 1.0:
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")

    if args.result_file:
        result_file = resolve_existing_path(args.result_file)
    else:
        result_file = find_default_result_file()
        if result_file is None:
            candidate_text = ", ".join(str(p) for p in DEFAULT_RESULT_CANDIDATES)
            raise FileNotFoundError(
                "No result file was provided and no default result file could be found. "
                f"Tried preferred candidates: {candidate_text}"
            )
    train_dir = resolve_existing_path(args.train_dir)
    test_image_dir = resolve_existing_path(args.test_image_dir)
    train_cache_path = resolve_existing_path(args.train_cache_path)
    test_cache_path = resolve_existing_path(args.test_cache_path)
    class_file = resolve_existing_path(args.class_file)
    output_dir = resolve_output_path(args.output_dir)
    outlier_dir = output_dir / "outlier_samples"

    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    if not train_cache_path.exists():
        raise FileNotFoundError(f"Training cache not found: {train_cache_path}")
    if not test_cache_path.exists():
        raise FileNotFoundError(f"Test cache not found: {test_cache_path}")

    predictions = parse_result_file(result_file)
    class_names = load_class_names(class_file)
    train_items, train_meta = load_yolo_cache(train_cache_path)
    test_items, test_meta = load_yolo_cache(test_cache_path)
    test_image_map = build_test_image_map(test_image_dir)
    test_cache_index = build_test_cache_index(test_items)

    class_stats_rows, class_stats = compute_class_area_stats(
        train_dir=train_dir,
        train_items=train_items,
        class_names=class_names,
        lower_quantile=float(args.lower_quantile),
        upper_quantile=float(args.upper_quantile),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outlier_dir.mkdir(parents=True, exist_ok=True)
    outliers, undetermined, analysis_summary = analyze_results(
        predictions=predictions,
        class_names=class_names,
        class_stats=class_stats,
        test_image_map=test_image_map,
        test_cache_index=test_cache_index,
        outlier_dir=outlier_dir,
    )

    class_stats_fields = [
        "class_name",
        "train_total_images",
        "valid_area_count",
        "missing_cache_count",
        "invalid_area_count",
        "too_few_samples",
        "min_area",
        "max_area",
        "mean_area",
        "median_area",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
        "lower_bound",
        "upper_bound",
    ]
    outlier_fields = [
        "image",
        "pred_label",
        "pred_class",
        "test_area_ratio",
        "train_p5",
        "train_p95",
        "train_median",
        "lower_bound",
        "upper_bound",
        "deviation_side",
        "source_image_path",
        "copied_path",
    ]
    undetermined_fields = [
        "image",
        "pred_label",
        "pred_class",
        "test_area_ratio",
        "source_image_path",
        "reason",
    ]

    write_csv(output_dir / "class_area_stats.csv", class_stats_rows, class_stats_fields)
    write_csv(output_dir / "outliers.csv", outliers, outlier_fields)
    write_csv(output_dir / "undetermined.csv", undetermined, undetermined_fields)

    write_json(output_dir / "class_area_stats.json", {"items": class_stats_rows})
    write_json(output_dir / "outliers.json", {"items": outliers})
    write_json(output_dir / "undetermined.json", {"items": undetermined})

    summary = {
        "result_file": str(result_file),
        "train_dir": str(train_dir),
        "test_image_dir": str(test_image_dir),
        "train_cache_path": str(train_cache_path),
        "test_cache_path": str(test_cache_path),
        "class_file": str(class_file),
        "output_dir": str(output_dir),
        "lower_quantile": float(args.lower_quantile),
        "upper_quantile": float(args.upper_quantile),
        "min_valid_class_samples": MIN_VALID_CLASS_SAMPLES,
        "train_cache_meta": train_meta,
        "test_cache_meta": test_meta,
        **analysis_summary,
    }
    write_json(output_dir / "summary.json", summary)

    print(f"Total predictions: {summary['total_predictions']}")
    print(f"Determinable samples: {summary['determinable_count']}")
    print(f"Outlier samples: {summary['outlier_count']}")
    print(f"Undetermined samples: {summary['undetermined_count']}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
