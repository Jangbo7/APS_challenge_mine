import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid, save_image

from augmentation import cutmix_data_yolo
from yolo_cutmix import YoloCutMixHelper


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(train_dir: str):
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    items = []
    for c in classes:
        cdir = os.path.join(train_dir, c)
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                items.append((os.path.join(cdir, fn), class_to_idx[c]))
    return items


def preprocess_resize_to_tensor(img_path: str, image_size: int) -> torch.Tensor:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im = im.resize((image_size, image_size), Image.BILINEAR)
    return TF.to_tensor(im)


def post_rgb_aug(
    images: torch.Tensor,
    scale_jitter: float = 0.05,
    noise_std: float = 0.015,
    sp_noise_p: float = 0.003,
) -> torch.Tensor:
    out = []
    for i in range(images.size(0)):
        x = images[i]
        _, h, w = x.shape

        # Mild scale jitter while keeping final tensor size unchanged.
        if scale_jitter > 0:
            scale = random.uniform(max(0.1, 1.0 - scale_jitter), 1.0 + scale_jitter)
            nh = max(1, int(round(h * scale)))
            nw = max(1, int(round(w * scale)))
            x_scaled = TF.resize(x, [nh, nw], interpolation=InterpolationMode.BILINEAR)
            if nh >= h and nw >= w:
                top = (nh - h) // 2
                left = (nw - w) // 2
                x = TF.crop(x_scaled, top, left, h, w)
            else:
                pad_t = (h - nh) // 2
                pad_b = h - nh - pad_t
                pad_l = (w - nw) // 2
                pad_r = w - nw - pad_l
                x = TF.pad(x_scaled, [pad_l, pad_t, pad_r, pad_b], fill=0.0)

        if random.random() < 0.5:
            x = TF.hflip(x)
        x = TF.rotate(
            x,
            angle=random.uniform(-15.0, 15.0),
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        x = TF.adjust_brightness(x, 1.0 + random.uniform(-0.2, 0.2))
        x = TF.adjust_contrast(x, 1.0 + random.uniform(-0.2, 0.2))
        x = TF.adjust_saturation(x, 1.0 + random.uniform(-0.2, 0.2))
        x = TF.adjust_hue(x, random.uniform(-0.1, 0.1))
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        if sp_noise_p > 0:
            p = max(0.0, min(1.0, sp_noise_p))
            rnd = torch.rand((h, w), device=x.device)
            salt = rnd < (0.5 * p)
            pepper = (rnd >= (0.5 * p)) & (rnd < p)
            if salt.any():
                x[:, salt] = 1.0
            if pepper.any():
                x[:, pepper] = 0.0
        out.append(torch.clamp(x, 0.0, 1.0))
    return torch.stack(out, dim=0)


def apply_non_eligible_routing(images, images_orig, lam, applied_mask, rand_idx, policy, mixup_alpha):
    non_eligible_mask = ~applied_mask
    n_non = int(non_eligible_mask.sum().item())
    route_mixup = 0
    route_none = 0

    if n_non > 0:
        if policy == "mixup":
            lam_non = float(np.random.beta(mixup_alpha, mixup_alpha)) if mixup_alpha > 0 else 1.0
            images[non_eligible_mask] = (
                lam_non * images_orig[non_eligible_mask]
                + (1.0 - lam_non) * images_orig[rand_idx[non_eligible_mask]]
            )
            lam[non_eligible_mask] = lam_non
            route_mixup = n_non
        else:
            images[non_eligible_mask] = images_orig[non_eligible_mask]
            lam[non_eligible_mask] = 1.0
            route_none = n_non

    return images, lam, route_mixup, route_none


def save_triplet(before, after_cutmix, after_post, save_path, n_show=8):
    n_show = min(n_show, before.size(0))
    rows = []
    for i in range(n_show):
        rows.extend([before[i], after_cutmix[i], after_post[i]])
    grid = make_grid(rows, nrow=3, padding=4)
    save_image(grid, save_path)


def main():
    parser = argparse.ArgumentParser(description="Simulate new augmentation order for yolo_cutmix.")
    parser.add_argument("--train-dir", type=str, default="c:\\Users\\jangb\\Desktop\\contest_group\\train_eff\\data\\APS_dataset\\train")
    parser.add_argument("--cache-path", type=str, default="eff/yolo_boxes_cache.json")
    parser.add_argument("--out-dir", type=str, default="eff/aug_sim_out")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=39)

    parser.add_argument("--alpha", type=float, default= 2)
    parser.add_argument("--min-box-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.6)
    parser.add_argument("--center-tolerance-ratio", type=float, default=0.10)
    parser.add_argument("--fallback", type=str, default="skip", choices=["skip", "random"])
    parser.add_argument("--key-mode", type=str, default="relative_to_train_dir", choices=["relative_to_train_dir", "absolute"])

    parser.add_argument("--post-scale-jitter", type=float, default=0.1)
    parser.add_argument("--post-noise-std", type=float, default=0.03)
    parser.add_argument("--post-sp-noise-p", type=float, default=0.005)

    parser.add_argument("--policy", type=str, default="none", choices=["mixup", "none"])
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--save-all-yolo-applied", type=int, default=1, choices=[0, 1])
    parser.add_argument("--save-all-yolo-dir", type=str, default="eff/aug_sim_out")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_all_yolo_applied = bool(args.save_all_yolo_applied)
    save_all_yolo_dir = None
    if save_all_yolo_applied:
        if args.save_all_yolo_dir:
            save_all_yolo_dir = Path(args.save_all_yolo_dir)
        else:
            save_all_yolo_dir = out_dir / "yolo_applied_samples"
        save_all_yolo_dir.mkdir(parents=True, exist_ok=True)

    items = list_images(args.train_dir)
    if len(items) < args.batch_size:
        raise RuntimeError(f"Not enough samples: {len(items)} < batch_size={args.batch_size}")

    random.shuffle(items)

    helper = YoloCutMixHelper(
        cache_path=args.cache_path,
        train_dir=args.train_dir,
        key_mode=args.key_mode,
        fallback_mode=args.fallback,
        min_box_area_ratio=args.min_box_area_ratio,
        max_box_area_ratio=args.max_box_area_ratio,
        center_tolerance_ratio=args.center_tolerance_ratio,
        debug_log=True,
    )

    ptr = 0
    rows = []
    summary = {
        "total": 0,
        "applied": 0,
        "skipped": 0,
        "skipped_missing": 0,
        "skipped_empty": 0,
        "skipped_invalid": 0,
        "skipped_too_small": 0,
        "skipped_too_large": 0,
        "skipped_dst_invalid": 0,
        "route_to_mixup": 0,
        "route_to_none": 0,
    }

    for b in range(args.num_batches):
        batch_items = items[ptr:ptr + args.batch_size]
        if len(batch_items) < args.batch_size:
            break
        ptr += args.batch_size

        img_paths = [p for p, _ in batch_items]
        labels = torch.tensor([y for _, y in batch_items], dtype=torch.long, device=device)

        images = torch.stack([preprocess_resize_to_tensor(p, args.image_size) for p in img_paths], dim=0).to(device)
        images_orig = images.clone()

        images, labels_a, labels_b, lam, stats, applied_mask, rand_idx = cutmix_data_yolo(
            images,
            labels,
            img_paths,
            helper,
            alpha=args.alpha,
        )

        if save_all_yolo_applied:
            for i in range(args.batch_size):
                if not bool(applied_mask[i].item()):
                    continue
                lam_i = float(lam[i].item()) if torch.is_tensor(lam) else float(lam)
                file_name = (
                    f"b{b:04d}_i{i:03d}_la{int(labels_a[i].item())}_"
                    f"lb{int(labels_b[i].item())}_lam{lam_i:.4f}.png"
                )
                save_image(images[i].detach().cpu(), str(save_all_yolo_dir / file_name))

        images, lam, r_mixup, r_none = apply_non_eligible_routing(
            images=images,
            images_orig=images_orig,
            lam=lam,
            applied_mask=applied_mask,
            rand_idx=rand_idx,
            policy=args.policy,
            mixup_alpha=args.mixup_alpha,
        )

        images_post = post_rgb_aug(
            images,
            scale_jitter=args.post_scale_jitter,
            noise_std=args.post_noise_std,
            sp_noise_p=args.post_sp_noise_p,
        )

        save_triplet(
            images_orig.detach().cpu(),
            images.detach().cpu(),
            images_post.detach().cpu(),
            str(out_dir / f"batch_{b:03d}_triplet.png"),
            n_show=min(8, args.batch_size),
        )

        lam_np = lam.detach().cpu().numpy().tolist()
        applied_np = applied_mask.detach().cpu().numpy().tolist()

        for i in range(args.batch_size):
            rows.append(
                {
                    "batch": b,
                    "index": i,
                    "img_path": img_paths[i],
                    "label_a": int(labels_a[i].item()),
                    "label_b": int(labels_b[i].item()),
                    "applied_cutmix_yolo": int(applied_np[i]),
                    "route": "cutmix_yolo" if applied_np[i] else f"non_eligible_{args.policy}",
                    "lam": float(lam_np[i]),
                }
            )

        for k in summary:
            if k in stats:
                summary[k] += int(stats[k])
        summary["route_to_mixup"] += r_mixup
        summary["route_to_none"] += r_none

        print(
            f"[Batch {b}] total={stats.get('total', 0)} applied={stats.get('applied', 0)} "
            f"skipped={stats.get('skipped', 0)} route_mixup={r_mixup} route_none={r_none} "
            f"lam_mean={float(lam.mean().item()):.4f}"
        )

    if rows:
        with open(out_dir / "sample_lam_report.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    total = max(1, summary["total"])
    summary["applied_ratio"] = summary["applied"] / total
    summary["skipped_ratio"] = summary["skipped"] / total
    summary["route_to_mixup_ratio"] = summary["route_to_mixup"] / total
    summary["route_to_none_ratio"] = summary["route_to_none"] / total

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done. Outputs:")
    print(out_dir / "sample_lam_report.csv")
    print(out_dir / "summary.json")
    if save_all_yolo_applied:
        print(save_all_yolo_dir)


if __name__ == "__main__":
    main()
