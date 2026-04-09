import csv
import os
from pathlib import Path

import torch
from tqdm import tqdm

from config4test2 import DualViewTestConfig
from dataset2 import get_dualview_test_dataloader, get_singleview_test_dataloader
from model import build_train2_model
from utils import load_checkpoint


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
    if getattr(config, "MODEL_VARIANT", "dual_branch") == "single_backbone_6ch":
        return torch.cat([raw_images, detail_images], dim=1), None
    return raw_images, detail_images


def prepare_single_view_input(images: torch.Tensor) -> torch.Tensor:
    return normalize_rgb_batch(images)


@torch.no_grad()
def test_dualview(model, test_loader, device, config):
    model.eval()
    results = []
    for raw_images, detail_images, rel_paths, _, _ in tqdm(test_loader, desc="[Testing2]"):
        raw_images = raw_images.to(device, non_blocking=True)
        detail_images = detail_images.to(device, non_blocking=True)
        model_input, detail_model_input = prepare_model_inputs(raw_images, detail_images, config)

        if detail_model_input is None:
            outputs = model(model_input)
        else:
            outputs = model(model_input, detail_model_input)
        logits = outputs["fusion_logits"]
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        confs = probs.max(dim=1).values

        for rel_path, pred, conf in zip(rel_paths, preds.cpu().numpy(), confs.cpu().numpy()):
            results.append((rel_path, int(pred), float(conf)))
    return results


@torch.no_grad()
def test_singleview(model, test_loader, device):
    model.eval()
    results = []
    for images, rel_paths, _ in tqdm(test_loader, desc="[Testing2-Single]"):
        images = images.to(device, non_blocking=True)
        model_input = prepare_single_view_input(images)
        outputs = model(model_input)
        logits = outputs["fusion_logits"]
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        confs = probs.max(dim=1).values

        for rel_path, pred, conf in zip(rel_paths, preds.cpu().numpy(), confs.cpu().numpy()):
            results.append((rel_path, int(pred), float(conf)))
    return results


def save_results_to_txt(results, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rel_path, pred, _ in results:
            f.write(f"{Path(rel_path).name} {pred}\n")
    print(f"Results saved to: {output_path}")


def save_results_to_csv(results, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "predicted_label", "confidence"])
        writer.writeheader()
        for rel_path, pred, conf in results:
            writer.writerow(
                {
                    "image": Path(rel_path).name,
                    "predicted_label": pred,
                    "confidence": conf,
                }
            )
    print(f"Detailed results saved to: {output_path}")


def main():
    config = DualViewTestConfig()
    train_mode = getattr(config, "TRAIN_MODE", "dual_view")
    single_view_source = getattr(config, "SINGLE_VIEW_SOURCE", "raw")
    if train_mode not in {"dual_view", "single_view"}:
        raise ValueError(f"Unknown TRAIN_MODE: {train_mode}")
    if train_mode == "single_view" and single_view_source not in {"raw", "detail"}:
        raise ValueError(f"Unknown SINGLE_VIEW_SOURCE: {single_view_source}")

    config.RAW_TEST_DIR = str(resolve_existing_path(config.RAW_TEST_DIR))
    config.DETAIL_TEST_DIR = str(resolve_existing_path(config.DETAIL_TEST_DIR))
    config.CLASSNAME_FILE = str(resolve_existing_path(config.CLASSNAME_FILE))
    config.CHECKPOINT_PATH = str(resolve_existing_path(config.CHECKPOINT_PATH))
    config.OUTPUT_FILE = str(resolve_output_path(config.OUTPUT_FILE))
    if getattr(config, "OUTPUT_CSV", None):
        config.OUTPUT_CSV = str(resolve_output_path(config.OUTPUT_CSV))
    if getattr(config, "PRETRAINED_WEIGHTS_DIR", None):
        config.PRETRAINED_WEIGHTS_DIR = str(resolve_output_path(config.PRETRAINED_WEIGHTS_DIR))

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(config.CHECKPOINT_PATH):
        raise FileNotFoundError(f"CHECKPOINT_PATH not found: {config.CHECKPOINT_PATH}")

    if train_mode == "single_view":
        test_loader, class_names = get_singleview_test_dataloader(config)
    else:
        test_loader, class_names = get_dualview_test_dataloader(config)
    print(f"Number of classes: {len(class_names)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    print("Building train2 model...")
    model = build_train2_model(config, pretrained=False)
    model = model.to(device)

    print(f"Loading checkpoint: {config.CHECKPOINT_PATH}")
    load_checkpoint(model, None, config.CHECKPOINT_PATH, device)

    if train_mode == "single_view":
        results = test_singleview(model, test_loader, device)
    else:
        results = test_dualview(model, test_loader, device, config)
    save_results_to_txt(results, config.OUTPUT_FILE)
    if getattr(config, "OUTPUT_CSV", None):
        save_results_to_csv(results, config.OUTPUT_CSV)

    print("\nFirst 10 predictions:")
    for rel_path, pred, _ in results[:10]:
        print(f"  {Path(rel_path).name} {pred}")


if __name__ == "__main__":
    main()
