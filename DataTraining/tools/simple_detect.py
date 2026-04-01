#!/usr/bin/env python3
"""
Simple object detection: just find bounding boxes, don't worry about class labels.
All detections saved as class 0 (rock/obstacle).
After training, the model learns what these boxes represent.
"""
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import cv2
except Exception:
    print("[ERROR] Missing dependency: cv2")
    raise SystemExit(1)

try:
    from ultralytics import YOLO
except Exception:
    print("[ERROR] Missing dependency: ultralytics")
    raise SystemExit(1)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
Box = Tuple[int, float, float, float, float, float]  # class_id, x1, y1, x2, y2, conf


def pixels_to_yolo(box: Box, img_w: int, img_h: int) -> str:
    """Convert pixel box to standard 5-column YOLO format."""
    cls_id, x1, y1, x2, y2, conf = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return f"{cls_id} {xc / img_w:.6f} {yc / img_h:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"


def save_yolo_labels(label_path: Path, boxes: List[Box], img_w: int, img_h: int) -> None:
    """Save labels in strict YOLO training format."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [pixels_to_yolo(box, img_w, img_h) for box in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple detection: save all detected boxes (class=0) for training"
    )
    parser.add_argument("--images-dir", default="DataTraining/data/raw_images", help="Images folder")
    parser.add_argument("--labels-dir", default="DataTraining/data/raw_labels", help="Labels folder")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model to use")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--output-dir", default="DataTraining/runs/simple_detect", help="Output dir")
    parser.add_argument("--save-json", action="store_true", help="Save JSON predictions")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    if not images_dir.exists():
        print(f"[ERROR] Images dir not found: {images_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    print(f"[INFO] Loaded model: {args.model}")
    print(f"[INFO] Mode: Simple detection (all objects as class 0)")

    image_paths = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not image_paths:
        print(f"[ERROR] No images found in {images_dir}")
        return 1

    print(f"\n[START] Processing {len(image_paths)} images...\n")

    total_boxes = 0

    for idx, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{idx}/{len(image_paths)}] [WARN] Failed to read: {image_path.name}")
            continue

        h, w = image.shape[:2]

        # Detect
        results = model.predict(source=image, conf=args.conf, iou=args.iou, verbose=False)
        if not results or not results[0].boxes:
            print(f"[{idx}/{len(image_paths)}] {image_path.name}: 0 boxes")
            continue

        # Convert all detections to class 0 (our generic obstacle class)
        boxes: List[Box] = []
        result = results[0]

        for det in result.boxes:
            x1, y1, x2, y2 = det.xyxy[0].tolist()
            conf = float(det.conf[0]) if hasattr(det, "conf") and len(det.conf) else 0.0
            # Save as class 0 - model will learn what this means
            boxes.append((0, float(x1), float(y1), float(x2), float(y2), conf))

        # Save labels
        label_path = labels_dir / f"{image_path.stem}.txt"
        save_yolo_labels(label_path, boxes, w, h)

        # Save JSON
        if args.save_json:
            json_path = output_dir / f"{image_path.stem}_detections.json"
            det_list = [
                {
                    "class": 0,
                    "class_name": "obstacle",
                    "confidence": float(boxes[i][5]),
                    "bbox_pixel": [float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3]), float(boxes[i][4])],
                }
                for i in range(len(boxes))
            ]
            json_path.write_text(json.dumps({"image": image_path.name, "detections": det_list}, indent=2))

        total_boxes += len(boxes)
        print(f"[{idx}/{len(image_paths)}] {image_path.name}: {len(boxes)} boxes")

    print(f"\n[DONE] Total boxes detected: {total_boxes}")
    print(f"  Labels saved to: {labels_dir}")
    print(f"  All boxes labeled as class 0 (obstacle)")
    print(f"  Next: Train model to classify what these obstacles are!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
