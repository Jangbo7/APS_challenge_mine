import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_IMAGE_DIR = Path("data/APS_dataset/test/test_noclass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two result txt files and export original images for samples with different predictions."
    )
    parser.add_argument("--result-a", default='eff/eff2/result.txt', help="Path to the first result txt file.")
    parser.add_argument("--result-b", default='eff/result.txt', help="Path to the second result txt file.")
    parser.add_argument(
        "--image-dir",
        default=str(DEFAULT_IMAGE_DIR),
        help="Directory containing original test images.",
    )
    parser.add_argument("--output-dir", default='eff/compare_f', help="Directory to store comparison outputs.")
    return parser.parse_args()


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

            filename, label_str = parts
            filename = Path(filename).name
            try:
                label = int(label_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label in {result_path} line {line_no}: {label_str!r} is not an integer"
                ) from exc

            if filename in predictions:
                raise ValueError(f"Duplicate filename in {result_path}: {filename}")

            predictions[filename] = label

    return predictions


def build_image_map(image_dir: Path) -> Dict[str, Path]:
    image_map: Dict[str, Path] = {}
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory is not a directory: {image_dir}")

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        name = image_path.name
        if name in image_map:
            raise ValueError(f"Duplicate image filename found in image dir: {name}")
        image_map[name] = image_path

    return image_map


def make_export_name(filename: str, pred_a: int, pred_b: int) -> str:
    source = Path(filename)
    suffix = source.suffix if source.suffix else ".jpg"
    stem = source.stem if source.stem else source.name
    return f"{stem}__a{pred_a}__b{pred_b}{suffix}"


def compare_predictions(
    preds_a: Dict[str, int],
    preds_b: Dict[str, int],
) -> Tuple[List[str], List[str], List[Tuple[str, int, int]], List[str]]:
    names_a = set(preds_a.keys())
    names_b = set(preds_b.keys())

    only_in_a = sorted(names_a - names_b)
    only_in_b = sorted(names_b - names_a)

    same_pred: List[str] = []
    different_pred: List[Tuple[str, int, int]] = []
    for filename in sorted(names_a & names_b):
        pred_a = preds_a[filename]
        pred_b = preds_b[filename]
        if pred_a == pred_b:
            same_pred.append(filename)
        else:
            different_pred.append((filename, pred_a, pred_b))

    return only_in_a, only_in_b, different_pred, same_pred


def write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_summary_txt(path: Path, summary: dict) -> None:
    lines = [
        f"result_a: {summary['result_a']}",
        f"result_b: {summary['result_b']}",
        f"image_dir: {summary['image_dir']}",
        f"output_dir: {summary['output_dir']}",
        f"count_a: {summary['count_a']}",
        f"count_b: {summary['count_b']}",
        f"intersection_count: {summary['intersection_count']}",
        f"same_pred_count: {summary['same_pred_count']}",
        f"different_pred_count: {summary['different_pred_count']}",
        f"only_in_a_count: {summary['only_in_a_count']}",
        f"only_in_b_count: {summary['only_in_b_count']}",
        f"copied_diff_images_count: {summary['copied_diff_images_count']}",
        f"missing_source_images_count: {summary['missing_source_images_count']}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    result_a_path = resolve_existing_path(args.result_a)
    result_b_path = resolve_existing_path(args.result_b)
    image_dir = resolve_existing_path(args.image_dir)
    output_dir = resolve_output_path(args.output_dir)
    diff_image_dir = output_dir / "different_images"

    if not result_a_path.exists():
        raise FileNotFoundError(f"Result file A not found: {result_a_path}")
    if not result_b_path.exists():
        raise FileNotFoundError(f"Result file B not found: {result_b_path}")

    preds_a = parse_result_file(result_a_path)
    preds_b = parse_result_file(result_b_path)
    image_map = build_image_map(image_dir)

    only_in_a, only_in_b, different_pred, same_pred = compare_predictions(preds_a, preds_b)

    output_dir.mkdir(parents=True, exist_ok=True)
    diff_image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "diff_predictions.csv"
    missing_images: List[dict] = []
    copied_count = 0

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "pred_a", "pred_b", "image_path", "copied_path"],
        )
        writer.writeheader()

        for filename, pred_a, pred_b in different_pred:
            image_path = image_map.get(filename)
            copied_path = ""
            image_path_str = ""

            if image_path is None:
                missing_images.append(
                    {
                        "filename": filename,
                        "pred_a": pred_a,
                        "pred_b": pred_b,
                        "reason": "source_image_not_found",
                    }
                )
            else:
                image_path_str = str(image_path)
                export_name = make_export_name(filename, pred_a, pred_b)
                export_path = diff_image_dir / export_name
                shutil.copy2(image_path, export_path)
                copied_path = str(export_path)
                copied_count += 1

            writer.writerow(
                {
                    "filename": filename,
                    "pred_a": pred_a,
                    "pred_b": pred_b,
                    "image_path": image_path_str,
                    "copied_path": copied_path,
                }
            )

    write_json(output_dir / "only_in_a.json", {"filenames": only_in_a})
    write_json(output_dir / "only_in_b.json", {"filenames": only_in_b})
    write_json(output_dir / "missing_source_images.json", {"items": missing_images})

    summary = {
        "result_a": str(result_a_path),
        "result_b": str(result_b_path),
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "count_a": len(preds_a),
        "count_b": len(preds_b),
        "intersection_count": len(preds_a.keys() & preds_b.keys()),
        "same_pred_count": len(same_pred),
        "different_pred_count": len(different_pred),
        "only_in_a_count": len(only_in_a),
        "only_in_b_count": len(only_in_b),
        "copied_diff_images_count": copied_count,
        "missing_source_images_count": len(missing_images),
    }
    write_summary_txt(output_dir / "summary.txt", summary)
    write_json(output_dir / "summary.json", summary)

    print(f"Result A count: {summary['count_a']}")
    print(f"Result B count: {summary['count_b']}")
    print(f"Intersection count: {summary['intersection_count']}")
    print(f"Same prediction count: {summary['same_pred_count']}")
    print(f"Different prediction count: {summary['different_pred_count']}")
    print(f"Only in A count: {summary['only_in_a_count']}")
    print(f"Only in B count: {summary['only_in_b_count']}")
    print(f"Copied different images count: {summary['copied_diff_images_count']}")
    print(f"Missing source images count: {summary['missing_source_images_count']}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
