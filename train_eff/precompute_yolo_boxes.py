import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from config4yolo import YoloPrecomputeConfig


def to_unix(path: str) -> str:
    return path.replace('\\', '/').strip()


def list_images(root: str) -> List[str]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                files.append(os.path.join(dp, fn))
    files.sort()
    return files


def build_key(img_path: str, train_dir: str, key_mode: str) -> str:
    if key_mode == 'relative_to_train_dir':
        key = os.path.relpath(img_path, train_dir)
    else:
        key = os.path.abspath(img_path)
    return to_unix(key)


def save_preview_image(
    img_path: str,
    boxes: List[List[float]],
    output_path: str,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    box_thickness: int = 2,
    text_color: Tuple[int, int, int] = (0, 0, 255),
):
    """保存带框的图片预览"""
    image = cv2.imread(img_path)
    if image is None:
        return

    # 绘制检测框
    for box in boxes:
        if len(box) >= 4:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            conf = float(box[4]) if len(box) > 4 else 0.0

            # 绘制框
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, box_thickness)

            # 绘制置信度
            text = f"{conf:.2f}"
            cv2.putText(
                image,
                text,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
            )

    # 保存
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)


def infer_boxes_for_image(model: YOLO, img_path: str, w: int, h: int, config, imgsz: int, conf: float) -> List[List[float]]:
    results = model.predict(
        source=img_path,
        conf=conf,
        max_det=config.MAX_DET,
        imgsz=imgsz,
        device=config.DEVICE,
        verbose=False,
    )

    boxes_out: List[List[float]] = []
    if results and len(results) > 0 and getattr(results[0], 'boxes', None) is not None:
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            c = float(confs[i]) if len(confs) > i else 0.0

            # clip to image bounds and keep valid boxes
            x1 = max(0.0, min(float(w - 1), x1))
            y1 = max(0.0, min(float(h - 1), y1))
            x2 = max(float(x1 + 1.0), min(float(w), x2))
            y2 = max(float(y1 + 1.0), min(float(h), y2))

            if x2 <= x1 or y2 <= y1:
                continue
            boxes_out.append([x1, y1, x2, y2, c])

    return boxes_out


def main():
    config = YoloPrecomputeConfig()

    train_dir = os.path.abspath(config.TRAIN_DIR)
    output = os.path.abspath(config.OUTPUT_CACHE_PATH)

    if os.path.exists(output) and not config.OVERWRITE:
        raise FileExistsError(f'Output exists: {output}. Set OVERWRITE=True in config4yolo.py to replace.')

    images = list_images(train_dir)
    if not images:
        raise RuntimeError(f'No images found under: {train_dir}')

    model = YOLO(config.MODEL)
    primary_imgsz = int(getattr(config, 'PRIMARY_IMGSZ', getattr(config, 'IMGSZ', 640)))
    primary_conf = float(getattr(config, 'CONF', 0.005))
    enable_second_pass = bool(getattr(config, 'ENABLE_SECOND_PASS', False))
    second_pass_imgsz = int(getattr(config, 'SECOND_PASS_IMGSZ', 1024))
    second_pass_conf = float(getattr(config, 'SECOND_PASS_CONF', primary_conf))
    second_pass_triggered = 0
    second_pass_recovered = 0

    items: Dict[str, List[List[float]]] = {}
    hit_images = 0
    empty_images = 0
    total_boxes = 0

    # 用于收集预览图的候选
    preview_with_boxes = []  # (img_path, boxes) 列表
    preview_without_boxes = []  # img_path 列表

    pbar = tqdm(images, desc='YOLO precompute')
    for img_path in pbar:
        key = build_key(img_path, train_dir, config.KEY_MODE)

        with Image.open(img_path) as im:
            w, h = im.size

        boxes_out = infer_boxes_for_image(model, img_path, w, h, config, primary_imgsz, primary_conf)
        if enable_second_pass and (not boxes_out) and second_pass_imgsz != primary_imgsz:
            second_pass_triggered += 1
            boxes_out = infer_boxes_for_image(model, img_path, w, h, config, second_pass_imgsz, second_pass_conf)
            if boxes_out:
                second_pass_recovered += 1

        items[key] = {
            'boxes': boxes_out,
            'orig_size': [int(w), int(h)],
        }
        total_boxes += len(boxes_out)
        if boxes_out:
            hit_images += 1
            if config.YOLO_PREVIEW_ENABLE:
                preview_with_boxes.append((img_path, boxes_out))
        else:
            empty_images += 1
            if config.YOLO_PREVIEW_ENABLE:
                preview_without_boxes.append(img_path)

    cache = {
        'meta': {
            'model': config.MODEL,
            'primary_conf': primary_conf,
            'max_det': config.MAX_DET,
            'device': config.DEVICE,
            'primary_imgsz': primary_imgsz,
            'enable_second_pass': enable_second_pass,
            'second_pass_imgsz': second_pass_imgsz if enable_second_pass else None,
            'second_pass_conf': second_pass_conf if enable_second_pass else None,
            'second_pass_triggered': second_pass_triggered,
            'second_pass_recovered': second_pass_recovered,
            'key_mode': config.KEY_MODE,
            'train_dir': to_unix(train_dir),
            'num_images': len(images),
            'box_coord_space': 'original_image',
        },
        'items': items,
    }

    Path(os.path.dirname(output) or '.').mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)

    avg_boxes = total_boxes / max(1, len(images))
    
    # 保存预览图
    if config.YOLO_PREVIEW_ENABLE:
        preview_dir = config.YOLO_PREVIEW_OUTPUT_DIR
        Path(preview_dir).mkdir(parents=True, exist_ok=True)

        saved_count = 0
        if config.YOLO_PREVIEW_BOX_TYPE in ['with_boxes', 'both']:
            for i, (img_path, boxes) in enumerate(preview_with_boxes[: config.YOLO_PREVIEW_NUM]):
                filename = f"with_boxes_{i:03d}_{os.path.basename(img_path)}"
                output_path = os.path.join(preview_dir, filename)
                save_preview_image(
                    img_path,
                    boxes,
                    output_path,
                    box_color=config.YOLO_PREVIEW_BOX_COLOR,
                    box_thickness=config.YOLO_PREVIEW_BOX_THICKNESS,
                    text_color=config.YOLO_PREVIEW_TEXT_COLOR,
                )
                saved_count += 1

        if config.YOLO_PREVIEW_BOX_TYPE in ['without_boxes', 'both']:
            for i, img_path in enumerate(preview_without_boxes[: config.YOLO_PREVIEW_NUM]):
                filename = f"without_boxes_{i:03d}_{os.path.basename(img_path)}"
                output_path = os.path.join(preview_dir, filename)
                image = cv2.imread(img_path)
                if image is not None:
                    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(output_path, image)
                    saved_count += 1

    print('\nDone.')
    print(f'Output: {output}')
    print(f'Total images: {len(images)}')
    print(f'Images with boxes: {hit_images}')
    print(f'Images without boxes: {empty_images}')
    print(f'Total boxes: {total_boxes}')
    print(f'Avg boxes/image: {avg_boxes:.3f}')
    if enable_second_pass:
        print(f'Second pass triggered: {second_pass_triggered}')
        print(f'Second pass recovered: {second_pass_recovered}')
    if config.YOLO_PREVIEW_ENABLE:
        print(f'Preview images saved to: {preview_dir} (count: {saved_count})')


if __name__ == '__main__':
    main()
