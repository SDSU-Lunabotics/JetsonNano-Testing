#!/usr/bin/env python3
import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import cv2
except Exception:
    print("[ERROR] Missing dependency: cv2")
    print("Install dependencies first:")
    print("  pip install -r DataTraining/requirements.txt")
    raise SystemExit(1)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

Box = Tuple[int, float, float, float, float]  # class_id, x1, y1, x2, y2 (pixel)

# Common arrow/enter keycodes across OpenCV backends.
RIGHT_KEYS = {83, 2555904, 65363, 63235}
LEFT_KEYS = {81, 2424832, 65361, 63234}
ENTER_KEYS = {10, 13}


def normalize_label(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_classes(path: Path) -> List[str]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    classes = [line for line in lines if line and not line.startswith("#")]
    if not classes:
        raise ValueError(f"No classes found in {path}")
    return classes


def yolo_to_pixels(label_line: str, img_w: int, img_h: int) -> Box:
    cls, xc, yc, w, h = label_line.strip().split()
    cls_id = int(cls)
    xc = float(xc) * img_w
    yc = float(yc) * img_h
    w = float(w) * img_w
    h = float(h) * img_h
    x1 = max(0.0, xc - w / 2.0)
    y1 = max(0.0, yc - h / 2.0)
    x2 = min(float(img_w - 1), xc + w / 2.0)
    y2 = min(float(img_h - 1), yc + h / 2.0)
    return (cls_id, x1, y1, x2, y2)


def pixels_to_yolo(box: Box, img_w: int, img_h: int) -> str:
    cls_id, x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return f"{cls_id} {xc / img_w:.6f} {yc / img_h:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"


def load_label_file(label_path: Path, img_w: int, img_h: int) -> List[Box]:
    if not label_path.exists():
        return []
    boxes: List[Box] = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            boxes.append(yolo_to_pixels(line, img_w, img_h))
        except Exception:
            pass
    return boxes


def save_label_file(label_path: Path, boxes: List[Box], img_w: int, img_h: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [pixels_to_yolo(box, img_w, img_h) for box in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def color_for_class(class_id: int) -> Tuple[int, int, int]:
    palette = [
        (0, 255, 0),
        (0, 180, 255),
        (255, 220, 0),
        (255, 120, 0),
        (180, 80, 255),
        (255, 0, 120),
    ]
    return palette[class_id % len(palette)]


def draw_overlay(
    image,
    boxes: List[Box],
    class_names: List[str],
    index: int,
    total: int,
    image_name: str,
    active_class_id: int,
    status_text: str,
):
    canvas = image.copy()
    cv2.putText(canvas, f"{index + 1}/{total} {image_name}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 255), 2)
    cv2.putText(
        canvas,
        "a:add box  d:auto  s:save  n/enter:next  p:left:prev  q:quit",
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (170, 170, 170),
        1,
    )

    active_name = class_names[active_class_id] if 0 <= active_class_id < len(class_names) else f"cls{active_class_id}"
    cv2.putText(
        canvas,
        f"Active class [{active_class_id}]: {active_name}  (press 0-9 to switch)",
        (10, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 255, 120),
        1,
    )

    if status_text:
        cv2.putText(canvas, status_text, (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)

    for i, (cls_id, x1, y1, x2, y2) in enumerate(boxes):
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls{cls_id}"
        color = color_for_class(cls_id)
        cv2.rectangle(canvas, p1, p2, color, 2)
        cv2.putText(
            canvas,
            f"{i}:{cls_name}",
            (p1[0], max(20, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return canvas


def build_help_panel(class_names: List[str], active_class_id: int, auto_enabled: bool) -> np.ndarray:
    panel = np.zeros((470, 560, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)

    y = 30
    cv2.putText(panel, "Annotator Help", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    y += 28
    cv2.putText(panel, "Keys", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (150, 200, 255), 2)
    y += 24

    lines = [
        "a : add box (drag with mouse)",
        "d : auto-detect current image",
        "r : remove last box",
        "u : undo last add batch",
        "c : clear all boxes",
        "s : save labels",
        "n or Enter or Right Arrow : next image",
        "p or Left Arrow : previous image",
        "0-9 : set active class",
        "q : save and quit",
    ]
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
        y += 22

    y += 8
    mode_text = "Auto-detect: ENABLED" if auto_enabled else "Auto-detect: DISABLED (no --auto-model)"
    cv2.putText(panel, mode_text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 120), 1)

    y += 30
    cv2.putText(panel, "Classes", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (150, 200, 255), 2)
    y += 24

    for idx, name in enumerate(class_names):
        color = color_for_class(idx)
        prefix = ">" if idx == active_class_id else " "
        cv2.putText(panel, f"{prefix} [{idx}] {name}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 1)
        y += 22

    return panel


def run_auto_detect(image, model, class_names: List[str], conf: float, iou: float, max_det: int):
    name_to_local: Dict[str, int] = {normalize_label(name): idx for idx, name in enumerate(class_names)}

    results = model.predict(source=image, conf=conf, iou=iou, max_det=max_det, verbose=False)
    if not results:
        return [], 0, 0

    out_boxes: List[Box] = []
    used = 0
    skipped = 0

    result = results[0]
    names = result.names if hasattr(result, "names") else {}

    if result.boxes is None:
        return [], 0, 0

    for det in result.boxes:
        cls_idx = int(det.cls[0])
        model_name = str(names.get(cls_idx, f"cls{cls_idx}"))
        local_idx = name_to_local.get(normalize_label(model_name))
        if local_idx is None:
            skipped += 1
            continue

        x1, y1, x2, y2 = det.xyxy[0].tolist()
        out_boxes.append((local_idx, float(x1), float(y1), float(x2), float(y2)))
        used += 1

    return out_boxes, used, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple YOLO image annotation tool")
    parser.add_argument("--images-dir", default="DataTraining/data/raw_images", help="Folder with images")
    parser.add_argument("--labels-dir", default="DataTraining/data/raw_labels", help="Folder for YOLO txt labels")
    parser.add_argument("--classes", default="DataTraining/classes.txt", help="Class names file")
    parser.add_argument("--auto-model", default=None, help="Optional YOLO model path for auto-detect")
    parser.add_argument("--auto-conf", type=float, default=0.25, help="Auto-detect confidence threshold")
    parser.add_argument("--auto-iou", type=float, default=0.45, help="Auto-detect IoU threshold")
    parser.add_argument("--auto-max-det", type=int, default=50, help="Auto-detect max detections")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    class_file = Path(args.classes)

    if not images_dir.exists():
        print(f"[ERROR] Missing images dir: {images_dir}")
        return 1
    if not class_file.exists():
        print(f"[ERROR] Missing classes file: {class_file}")
        return 1

    class_names = load_classes(class_file)
    image_paths = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

    if not image_paths:
        print(f"[ERROR] No images found in {images_dir}")
        return 1

    auto_model = None
    if args.auto_model:
        try:
            from ultralytics import YOLO
        except Exception:
            print("[ERROR] Auto-detect requires ultralytics")
            print("Install dependencies first:")
            print("  pip install -r DataTraining/requirements.txt")
            return 1

        auto_model = YOLO(args.auto_model)
        print(f"[INFO] Loaded auto model: {args.auto_model}")

    cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Annotator Help", cv2.WINDOW_NORMAL)

    idx = 0
    active_class_id = 0
    cache: Dict[Path, List[Box]] = {}
    action_history: Dict[Path, List[int]] = {}

    status_text = ""
    status_until = 0.0

    def set_status(msg: str, seconds: float = 2.0) -> None:
        nonlocal status_text, status_until
        status_text = msg
        status_until = time.time() + seconds

    while 0 <= idx < len(image_paths):
        image_path = image_paths[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Failed to read image: {image_path}")
            idx += 1
            continue

        h, w = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"

        if image_path not in cache:
            cache[image_path] = load_label_file(label_path, w, h)
        if image_path not in action_history:
            action_history[image_path] = []

        boxes: List[Box] = cache[image_path]

        while True:
            msg = status_text if time.time() < status_until else ""
            view = draw_overlay(
                image,
                boxes,
                class_names,
                idx,
                len(image_paths),
                image_path.name,
                active_class_id,
                msg,
            )
            help_panel = build_help_panel(class_names, active_class_id, auto_enabled=auto_model is not None)

            cv2.imshow("Annotator", view)
            cv2.imshow("Annotator Help", help_panel)

            key = cv2.waitKeyEx(30)
            if key < 0:
                continue

            ch = key & 0xFF

            if ord("0") <= ch <= ord("9"):
                chosen = ch - ord("0")
                if 0 <= chosen < len(class_names):
                    active_class_id = chosen
                    set_status(f"Active class set to [{active_class_id}] {class_names[active_class_id]}")
                else:
                    set_status(f"Class [{chosen}] does not exist")
                continue

            if ch in (ord("a"), ord("A")):
                roi = cv2.selectROI("Annotator", image, fromCenter=False, showCrosshair=True)
                x, y, rw, rh = roi
                if rw > 1 and rh > 1:
                    boxes.append((active_class_id, float(x), float(y), float(x + rw), float(y + rh)))
                    action_history[image_path].append(1)
                    set_status(f"Added 1 box as {class_names[active_class_id]}")
                else:
                    set_status("Box canceled")
            elif ch in (ord("d"), ord("D")):
                if auto_model is None:
                    set_status("Auto-detect disabled. Restart with --auto-model")
                    continue

                auto_boxes, used, skipped = run_auto_detect(
                    image,
                    auto_model,
                    class_names,
                    conf=args.auto_conf,
                    iou=args.auto_iou,
                    max_det=max(1, args.auto_max_det),
                )
                if auto_boxes:
                    boxes.extend(auto_boxes)
                    action_history[image_path].append(len(auto_boxes))
                    set_status(f"Auto-detect added {used} boxes (skipped {skipped} unmapped)")
                else:
                    set_status("Auto-detect found no mapped detections")
            elif ch in (ord("r"), ord("R")):
                if boxes:
                    boxes.pop()
                    set_status("Removed last box")
                else:
                    set_status("No boxes to remove")
            elif ch in (ord("u"), ord("U")):
                if action_history[image_path]:
                    count = action_history[image_path].pop()
                    if count > 0 and boxes:
                        del boxes[max(0, len(boxes) - count):]
                        set_status(f"Undid last batch ({count} boxes)")
                else:
                    set_status("Nothing to undo")
            elif ch in (ord("c"), ord("C")):
                boxes.clear()
                action_history[image_path].clear()
                set_status("Cleared all boxes")
            elif ch in (ord("s"), ord("S")):
                save_label_file(label_path, boxes, w, h)
                set_status(f"Saved {len(boxes)} boxes")
                print(f"[SAVE] {label_path}")
            elif ch in (ord("n"), ord("N")) or key in RIGHT_KEYS or key in ENTER_KEYS:
                save_label_file(label_path, boxes, w, h)
                if idx >= len(image_paths) - 1:
                    set_status("Already at last image")
                else:
                    idx += 1
                    break
            elif ch in (ord("p"), ord("P")) or key in LEFT_KEYS:
                save_label_file(label_path, boxes, w, h)
                if idx <= 0:
                    set_status("Already at first image")
                else:
                    idx -= 1
                    break
            elif ch in (ord("q"), ord("Q")):
                save_label_file(label_path, boxes, w, h)
                cv2.destroyAllWindows()
                print("[DONE] Annotation session ended")
                return 0

    cv2.destroyAllWindows()
    print("[DONE] Annotation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
