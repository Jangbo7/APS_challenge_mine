import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _to_unix(path: str) -> str:
    return path.replace('\\', '/').strip()


def _abs_norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


@dataclass
class YoloCutMixStats:
    total: int = 0
    applied: int = 0
    skipped_missing: int = 0
    skipped_empty: int = 0
    skipped_invalid: int = 0
    skipped_too_small: int = 0
    skipped_too_large: int = 0
    skipped_overlap: int = 0
    skipped_dst_invalid: int = 0
    pair_random: int = 0
    pair_threshold_matched: int = 0
    pair_threshold_fallback_random: int = 0
    pair_no_area_fallback: int = 0
    pair_ratio_min_current: float = 0.0
    pair_ratio_max_current: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            'total': self.total,
            'applied': self.applied,
            'skipped_missing': self.skipped_missing,
            'skipped_empty': self.skipped_empty,
            'skipped_invalid': self.skipped_invalid,
            'skipped_too_small': self.skipped_too_small,
            'skipped_too_large': self.skipped_too_large,
            'skipped_overlap': self.skipped_overlap,
            'skipped_dst_invalid': self.skipped_dst_invalid,
            'pair_random': self.pair_random,
            'pair_threshold_matched': self.pair_threshold_matched,
            'pair_threshold_fallback_random': self.pair_threshold_fallback_random,
            'pair_no_area_fallback': self.pair_no_area_fallback,
            'pair_ratio_min_current': self.pair_ratio_min_current,
            'pair_ratio_max_current': self.pair_ratio_max_current,
            'skipped': (
                self.skipped_missing
                + self.skipped_empty
                + self.skipped_invalid
                + self.skipped_overlap
                + self.skipped_dst_invalid
            ),
        }


class YoloCutMixHelper:
    def __init__(
        self,
        cache_path: str,
        train_dir: str,
        key_mode: str = 'relative_to_train_dir',
        fallback_mode: str = 'skip',
        min_box_area_ratio: float = 0.001,
        max_box_area_ratio: float = 0.8,
        prob_in_box: float = 0.7,
        center_tolerance_ratio: float = 0.10,
        debug_log: bool = True,
        pair_random_prob: float = 0.2,
        pair_area_ratio_min: float = 0.2,
        pair_area_ratio_max: float = 5.0,
        pair_area_ratio_min_target: float = 0.3,
        pair_area_ratio_max_target: float = 3.0,
        pair_schedule_start_ratio: float = 0.30,
        pair_schedule_end_ratio: float = 0.85,
    ):
        self.cache_path = cache_path
        self.train_dir = _abs_norm(train_dir)
        self.key_mode = key_mode
        self.fallback_mode = fallback_mode
        self.min_box_area_ratio = float(min_box_area_ratio)
        self.max_box_area_ratio = float(max_box_area_ratio)
        self.prob_in_box = float(prob_in_box)
        self.center_tolerance_ratio = float(center_tolerance_ratio)
        self.debug_log = bool(debug_log)
        self.pair_random_prob = float(np.clip(pair_random_prob, 0.0, 1.0))
        self.pair_area_ratio_min = float(max(1e-6, pair_area_ratio_min))
        self.pair_area_ratio_max = float(max(self.pair_area_ratio_min, pair_area_ratio_max))
        self.pair_area_ratio_min_target = float(max(1e-6, pair_area_ratio_min_target))
        self.pair_area_ratio_max_target = float(max(self.pair_area_ratio_min_target, pair_area_ratio_max_target))
        self.pair_schedule_start_ratio = float(np.clip(pair_schedule_start_ratio, 0.0, 1.0))
        self.pair_schedule_end_ratio = float(np.clip(pair_schedule_end_ratio, 0.0, 1.0))
        if self.pair_schedule_end_ratio < self.pair_schedule_start_ratio:
            self.pair_schedule_end_ratio = self.pair_schedule_start_ratio

        self.current_pair_area_ratio_min = self.pair_area_ratio_min
        self.current_pair_area_ratio_max = self.pair_area_ratio_max

        self.meta: Dict[str, object] = {}
        self.items: Dict[str, object] = {}
        self._warned_missing_orig_size = False
        self._grid_cache: Dict[Tuple[str, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._load_cache()

    def set_pair_schedule_progress(self, progress: float):
        """根据训练进度更新面积比匹配窗口。progress in [0, 1]."""
        p = float(np.clip(progress, 0.0, 1.0))
        s = self.pair_schedule_start_ratio
        e = self.pair_schedule_end_ratio

        if p <= s or e <= s:
            t = 0.0
        elif p >= e:
            t = 1.0
        else:
            t = (p - s) / max(e - s, 1e-12)

        self.current_pair_area_ratio_min = (1.0 - t) * self.pair_area_ratio_min + t * self.pair_area_ratio_min_target
        self.current_pair_area_ratio_max = (1.0 - t) * self.pair_area_ratio_max + t * self.pair_area_ratio_max_target
        if self.current_pair_area_ratio_max < self.current_pair_area_ratio_min:
            self.current_pair_area_ratio_max = self.current_pair_area_ratio_min

    def _load_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            if self.debug_log:
                print(f"[YOLO-CutMix] Cache not found: {self.cache_path}")
            self.meta = {}
            self.items = {}
            return

        with open(self.cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'items' in data:
            self.meta = data.get('meta', {}) if isinstance(data.get('meta', {}), dict) else {}
            raw_items = data.get('items', {})
        else:
            self.meta = {}
            raw_items = data

        if not isinstance(raw_items, dict):
            raise ValueError(f"Invalid YOLO cache format: items should be dict, got {type(raw_items)}")

        self.items = {_to_unix(k): v for k, v in raw_items.items()}
        if self.debug_log:
            print(f"[YOLO-CutMix] Loaded cache entries: {len(self.items)} from {self.cache_path}")

    def _candidate_keys(self, img_path: str) -> List[str]:
        keys = []
        p_unix = _to_unix(img_path)
        keys.append(p_unix)

        img_abs = _abs_norm(img_path)
        img_abs_unix = _to_unix(img_abs)
        keys.append(img_abs_unix)

        if self.key_mode == 'relative_to_train_dir':
            try:
                rel = os.path.relpath(img_abs, self.train_dir)
                keys.append(_to_unix(rel))
            except Exception:
                pass

        # fallback: locate '/train/' segment if present in key
        marker = '/train/'
        if marker in p_unix:
            keys.append(p_unix.split(marker, 1)[1])
        if marker in img_abs_unix:
            keys.append(img_abs_unix.split(marker, 1)[1])

        # deduplicate while preserving order
        seen = set()
        out = []
        for k in keys:
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def _lookup_entry(self, img_path: str) -> Optional[object]:
        for key in self._candidate_keys(img_path):
            if key in self.items:
                return self.items[key]
        return None

    def _parse_entry(self, entry: object) -> Tuple[List[List[float]], Optional[Tuple[int, int]]]:
        # Backward compatible:
        # old format: entry is List[List[float]]
        # new format: entry is {'boxes': [...], 'orig_size': [w, h]}
        if entry is None:
            return [], None

        if isinstance(entry, list):
            return entry, None

        if isinstance(entry, dict):
            boxes = entry.get('boxes', [])
            orig_size = entry.get('orig_size')
            if (
                isinstance(orig_size, (list, tuple))
                and len(orig_size) >= 2
                and float(orig_size[0]) > 0
                and float(orig_size[1]) > 0
            ):
                return boxes if isinstance(boxes, list) else [], (int(orig_size[0]), int(orig_size[1]))
            return boxes if isinstance(boxes, list) else [], None

        return [], None

    def _map_box_to_current_size(
        self,
        box: Sequence[float],
        orig_size: Optional[Tuple[int, int]],
        h: int,
        w: int,
    ) -> Sequence[float]:
        if orig_size is None:
            if self.debug_log and not self._warned_missing_orig_size:
                print("[YOLO-CutMix] Warning: cache entry missing orig_size, boxes are used without coordinate scaling.")
                self._warned_missing_orig_size = True
            return box

        ow, oh = orig_size
        if ow <= 0 or oh <= 0:
            return box

        sx = float(w) / float(ow)
        sy = float(h) / float(oh)
        mapped = list(box)
        if len(mapped) >= 4:
            mapped[0] = float(mapped[0]) * sx
            mapped[1] = float(mapped[1]) * sy
            mapped[2] = float(mapped[2]) * sx
            mapped[3] = float(mapped[3]) * sy
        return mapped

    def _clip_box(
        self,
        box: Sequence[float],
        h: int,
        w: int,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[str]]:
        if len(box) < 4:
            return None, 'invalid'

        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None, 'invalid'

        area_ratio = (x2 - x1) * (y2 - y1) / float(h * w)
        if area_ratio < self.min_box_area_ratio or area_ratio > self.max_box_area_ratio:
            if area_ratio < self.min_box_area_ratio:
                return None, 'too_small'
            return None, 'too_large'

        return (x1, y1, x2, y2), None

    def _sample_patch_within_box(
        self,
        box_xyxy: Tuple[int, int, int, int],
        target_area: int,
        h: int,
        w: int,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box_xyxy
        bw = x2 - x1
        bh = y2 - y1

        if random.random() >= self.prob_in_box:
            return x1, y1, x2, y2

        patch_h = max(1, min(bh, int(np.sqrt(max(1, target_area)))))
        patch_w = max(1, min(bw, int(max(1, target_area) / patch_h)))

        # try to fit patch inside detection box
        if patch_h > bh:
            patch_h = bh
        if patch_w > bw:
            patch_w = bw

        if patch_h <= 0 or patch_w <= 0:
            return x1, y1, x2, y2

        sx = random.randint(x1, x2 - patch_w)
        sy = random.randint(y1, y2 - patch_h)
        ex = min(w, sx + patch_w)
        ey = min(h, sy + patch_h)

        if ex <= sx or ey <= sy:
            return x1, y1, x2, y2

        return sx, sy, ex, ey

    @staticmethod
    def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def _compute_recenter_shift(
        self,
        box: Tuple[int, int, int, int],
        h: int,
        w: int,
    ) -> Tuple[int, int]:
        cx, cy = self._box_center(box)
        img_cx = 0.5 * (w - 1)
        img_cy = 0.5 * (h - 1)
        dist = float(np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2))
        tol = self.center_tolerance_ratio * float(min(h, w))
        if dist <= tol:
            return 0, 0
        return int(round(img_cx - cx)), int(round(img_cy - cy))

    @staticmethod
    def _translate_image(img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        if dx == 0 and dy == 0:
            return img

        c, h, w = img.shape
        out = torch.zeros_like(img)

        src_x1 = max(0, -dx)
        src_x2 = min(w, w - dx)
        dst_x1 = max(0, dx)
        dst_x2 = min(w, w + dx)

        src_y1 = max(0, -dy)
        src_y2 = min(h, h - dy)
        dst_y1 = max(0, dy)
        dst_y2 = min(h, h + dy)

        if src_x2 <= src_x1 or src_y2 <= src_y1 or dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return out

        out[:, dst_y1:dst_y2, dst_x1:dst_x2] = img[:, src_y1:src_y2, src_x1:src_x2]
        return out

    @staticmethod
    def _translate_box(
        box: Tuple[int, int, int, int],
        dx: int,
        dy: int,
        h: int,
        w: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = box
        x1 += dx
        x2 += dx
        y1 += dy
        y2 += dy

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _build_sector_mask(
        self,
        h: int,
        w: int,
        theta_start: float,
        theta_end: float,
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = (str(device), h, w)
        cached = self._grid_cache.get(cache_key)
        if cached is None:
            yy = torch.arange(h, device=device, dtype=torch.float32).view(h, 1).expand(h, w)
            xx = torch.arange(w, device=device, dtype=torch.float32).view(1, w).expand(h, w)
            self._grid_cache[cache_key] = (yy, xx)
        else:
            yy, xx = cached

        cx = 0.5 * (w - 1)
        cy = 0.5 * (h - 1)

        ang = torch.atan2(yy - cy, xx - cx)
        two_pi = 2.0 * np.pi
        ang = torch.remainder(ang + two_pi, two_pi)
        s = float((theta_start + two_pi) % two_pi)
        e = float((theta_end + two_pi) % two_pi)

        if s <= e:
            mask = (ang >= s) & (ang <= e)
        else:
            mask = (ang >= s) | (ang <= e)
        return mask

    def _build_valid_boxes(
        self,
        boxes_raw: object,
        orig_size: Optional[Tuple[int, int]],
        h: int,
        w: int,
    ) -> List[Tuple[int, int, int, int]]:
        valid_boxes: List[Tuple[int, int, int, int]] = []
        if not isinstance(boxes_raw, list):
            return valid_boxes
        for b in boxes_raw:
            mapped_b = self._map_box_to_current_size(b, orig_size, h=h, w=w)
            clipped, _ = self._clip_box(mapped_b, h=h, w=w)
            if clipped is not None:
                valid_boxes.append(clipped)
        return valid_boxes

    def _estimate_area_ratio_from_valid_boxes(
        self,
        valid_boxes: List[Tuple[int, int, int, int]],
        h: int,
        w: int,
    ) -> Optional[float]:
        if not valid_boxes:
            return None
        denom = float(h * w)
        area_ratios = [((x2 - x1) * (y2 - y1)) / denom for (x1, y1, x2, y2) in valid_boxes]
        return float(np.median(area_ratios)) if area_ratios else None

    def _estimate_min_area_ratio_from_valid_boxes(
        self,
        valid_boxes: List[Tuple[int, int, int, int]],
        h: int,
        w: int,
    ) -> Optional[float]:
        if not valid_boxes:
            return None
        denom = float(h * w)
        area_ratios = [((x2 - x1) * (y2 - y1)) / denom for (x1, y1, x2, y2) in valid_boxes]
        return float(min(area_ratios)) if area_ratios else None

    def _estimate_entry_area_ratio(
        self,
        entry: Optional[object],
        h: int,
        w: int,
    ) -> Optional[float]:
        boxes_raw, orig_size = self._parse_entry(entry)
        if not isinstance(boxes_raw, list) or len(boxes_raw) == 0:
            return None

        area_ratios = []
        denom = float(h * w)
        for b in boxes_raw:
            mapped_b = self._map_box_to_current_size(b, orig_size, h=h, w=w)
            clipped, _ = self._clip_box(mapped_b, h=h, w=w)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            area_ratios.append(((x2 - x1) * (y2 - y1)) / denom)

        if not area_ratios:
            return None
        return float(np.median(area_ratios))

    def _build_pair_indices(
        self,
        area_ratios: List[Optional[float]],
        device: torch.device,
        stats: YoloCutMixStats,
    ) -> torch.Tensor:
        bsz = len(area_ratios)
        rand_idx = torch.randperm(bsz, device=device)
        if bsz <= 1:
            stats.pair_random = bsz
            return rand_idx

        min_r = self.current_pair_area_ratio_min
        max_r = self.current_pair_area_ratio_max

        for i in range(bsz):
            if random.random() < self.pair_random_prob:
                stats.pair_random += 1
                continue

            a_i = area_ratios[i]
            if a_i is None or a_i <= 0:
                stats.pair_no_area_fallback += 1
                stats.pair_threshold_fallback_random += 1
                continue

            candidates = []
            for j in range(bsz):
                if j == i:
                    continue
                a_j = area_ratios[j]
                if a_j is None or a_j <= 0:
                    continue
                ratio = a_j / a_i
                if min_r <= ratio <= max_r:
                    candidates.append(j)

            if candidates:
                rand_idx[i] = int(random.choice(candidates))
                stats.pair_threshold_matched += 1
            else:
                stats.pair_threshold_fallback_random += 1

        return rand_idx

    def apply(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        img_paths: Sequence[str],
        alpha: float = 1.0,
    ):
        bsz, _, h, w = images.shape
        stats = YoloCutMixStats(total=bsz)

        # 批内预解析，避免同一 batch 多次查找/解析缓存
        entries: List[Optional[object]] = []
        parsed_entries: List[Tuple[List[List[float]], Optional[Tuple[int, int]]]] = []
        valid_boxes_all: List[List[Tuple[int, int, int, int]]] = []
        area_ratios = []
        area_ratios_min = []
        for path in img_paths:
            entry = self._lookup_entry(path)
            boxes_raw, orig_size = self._parse_entry(entry)
            valid_boxes = self._build_valid_boxes(boxes_raw, orig_size, h=h, w=w)

            entries.append(entry)
            parsed_entries.append((boxes_raw, orig_size))
            valid_boxes_all.append(valid_boxes)
            area_ratios.append(self._estimate_area_ratio_from_valid_boxes(valid_boxes, h=h, w=w))
            area_ratios_min.append(self._estimate_min_area_ratio_from_valid_boxes(valid_boxes, h=h, w=w))

        rand_idx = self._build_pair_indices(area_ratios, device=images.device, stats=stats)
        stats.pair_ratio_min_current = float(self.current_pair_area_ratio_min)
        stats.pair_ratio_max_current = float(self.current_pair_area_ratio_max)

        if alpha > 0:
            lam_scalar = float(np.random.beta(alpha, alpha))
        else:
            lam_scalar = 1.0

        # lam_scalar controls angular span ratio, actual lam uses exact mask pixel ratio.
        span_ratio = float(np.clip(1.0 - lam_scalar, 0.0, 1.0))
        span_angle = span_ratio * 2.0 * np.pi

        mixed_images = images.clone()
        labels_a = labels
        labels_b = labels[rand_idx]
        lam_batch = torch.ones(bsz, dtype=images.dtype, device=images.device)
        applied_mask = torch.zeros(bsz, dtype=torch.bool, device=images.device)

        for i in range(bsz):
            dst_boxes = valid_boxes_all[i]

            # 目标样本面积过滤硬门槛：目标图无有效面积框则不执行 cutmix_yolo
            if len(dst_boxes) == 0:
                stats.skipped_dst_invalid += 1
                continue

            # 目标框用于中心偏移校正（若偏离中心超过阈值，先平移到中心）
            dst_box = random.choice(dst_boxes)
            dst_dx, dst_dy = self._compute_recenter_shift(dst_box, h=h, w=w)
            dst_img = mixed_images[i]
            if dst_dx != 0 or dst_dy != 0:
                dst_img = self._translate_image(dst_img, dst_dx, dst_dy)
                moved_dst_box = self._translate_box(dst_box, dst_dx, dst_dy, h=h, w=w)
                if moved_dst_box is None:
                    stats.skipped_dst_invalid += 1
                    continue
                dst_box = moved_dst_box

            src_i = int(rand_idx[i].item())
            src_entry = entries[src_i]
            src_boxes_raw, src_orig_size = parsed_entries[src_i]

            if src_entry is None:
                stats.skipped_missing += 1
                if self.fallback_mode == 'random':
                    # 没有源框时退化：仍使用中心扇形同位替换（源不做中心校正）
                    src_img = images[src_i]
                    theta_start = float(np.random.uniform(-np.pi, np.pi))
                    theta_end = theta_start + span_angle
                    mask = self._build_sector_mask(h=h, w=w, theta_start=theta_start, theta_end=theta_end, device=images.device)
                    mask3 = mask.unsqueeze(0).expand_as(dst_img)
                    dst_img = torch.where(mask3, src_img, dst_img)
                    mixed_images[i] = dst_img
                    lam_batch[i] = 1.0 - float(mask.float().mean().item())
                    applied_mask[i] = True
                    stats.applied += 1
                continue

            if not isinstance(src_boxes_raw, list) or len(src_boxes_raw) == 0:
                stats.skipped_empty += 1
                continue

            src_boxes = valid_boxes_all[src_i]
            if len(src_boxes) == 0:
                stats.skipped_invalid += 1
                continue

            clipped = random.choice(src_boxes)
            reason = None
            if clipped is None:
                if reason == 'too_small':
                    stats.skipped_too_small += 1
                elif reason == 'too_large':
                    stats.skipped_too_large += 1
                else:
                    stats.skipped_invalid += 1
                continue

            # 源框同样用于中心偏移校正
            src_dx, src_dy = self._compute_recenter_shift(clipped, h=h, w=w)
            src_img = images[src_i]
            if src_dx != 0 or src_dy != 0:
                src_img = self._translate_image(src_img, src_dx, src_dy)
                moved_src_box = self._translate_box(clipped, src_dx, src_dy, h=h, w=w)
                if moved_src_box is None:
                    stats.skipped_invalid += 1
                    continue

            theta_start = float(np.random.uniform(-np.pi, np.pi))
            theta_end = theta_start + span_angle
            mask = self._build_sector_mask(h=h, w=w, theta_start=theta_start, theta_end=theta_end, device=images.device)
            mask3 = mask.unsqueeze(0).expand_as(dst_img)
            dst_img = torch.where(mask3, src_img, dst_img)

            mixed_images[i] = dst_img
            lam_batch[i] = 1.0 - float(mask.float().mean().item())
            applied_mask[i] = True
            stats.applied += 1

        stats_dict = stats.to_dict()
        # 透传每样本面积比：中位值给配对统计，最小值给后置缩放决策。
        stats_dict['area_ratios'] = area_ratios
        stats_dict['area_ratios_min'] = area_ratios_min
        return mixed_images, labels_a, labels_b, lam_batch, stats_dict, applied_mask, rand_idx
