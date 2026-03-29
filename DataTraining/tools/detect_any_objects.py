#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    print("[ERROR] Missing dependency: cv2")
    print("Install dependencies first:")
    print("  pip install -r DataTraining/requirements.txt")
    raise SystemExit(1)

try:
    from ultralytics import YOLO
except Exception:
    print("[ERROR] Missing dependency: ultralytics")
    print("Install dependencies first:")
    print("  pip install -r DataTraining/requirements.txt")
    raise SystemExit(1)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm"}


def normalize_label(text: str) -> str:
    return "_".join(text.strip().lower().replace("-", " ").split())


def parse_items(comma_text: str) -> List[str]:
    items = [x.strip() for x in comma_text.split(",")]
    return [x for x in items if x]


def read_lines(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    lines = [line.strip() for line in p.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def build_set(items: Iterable[str]) -> set:
    return {normalize_label(x) for x in items if x.strip()}


def draw_box(frame, xyxy: Tuple[int, int, int, int], text: str, color=(0, 255, 0)) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def keep_detection(
    label: str,
    box_xyxy: Tuple[float, float, float, float],
    frame_shape,
    ignore_labels: set,
    min_bottom_frac: float,
    min_box_area_px: int,
) -> bool:
    nlabel = normalize_label(label)
    if nlabel in ignore_labels:
        return False

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(w - 1), x2)
    y2 = min(float(h - 1), y2)

    if (x2 - x1) * (y2 - y1) < float(max(1, min_box_area_px)):
        return False

    # Heuristic: keep mostly objects close to the ground plane in the image.
    if h > 0 and (y2 / float(h)) < float(min_bottom_frac):
        return False

    return True


def maybe_map_to_rock(label: str, rock_aliases: set, force_rock: bool) -> str:
    if not force_rock:
        return label
    nlabel = normalize_label(label)
    if nlabel in rock_aliases:
        return "rock"
    return label


def run_inference_on_frame(
    model,
    frame,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    ignore_labels: set,
    min_bottom_frac: float,
    min_box_area_px: int,
    rock_aliases: set,
    force_rock: bool,
):
    result = model.predict(
        source=frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]

    names = result.names if hasattr(result, "names") else {}
    kept = []

    if result.boxes is None:
        return kept

    for det in result.boxes:
        cls_id = int(det.cls[0])
        score = float(det.conf[0]) if hasattr(det, "conf") and len(det.conf) else 0.0
        raw_label = str(names.get(cls_id, f"cls{cls_id}"))
        x1, y1, x2, y2 = [float(v) for v in det.xyxy[0].tolist()]

        if not keep_detection(
            raw_label,
            (x1, y1, x2, y2),
            frame.shape,
            ignore_labels,
            min_bottom_frac,
            min_box_area_px,
        ):
            continue

        out_label = maybe_map_to_rock(raw_label, rock_aliases, force_rock)
        kept.append({
            "label": out_label,
            "score": score,
            "xyxy": [x1, y1, x2, y2],
        })

    return kept


def process_image(
    image_path: Path,
    model,
    args,
    ignore_labels: set,
    rock_aliases: set,
    out_dir: Path,
):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] Could not read image: {image_path}")
        return 0

    detections = run_inference_on_frame(
        model,
        image,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        ignore_labels=ignore_labels,
        min_bottom_frac=args.min_bottom_frac,
        min_box_area_px=args.min_box_area_px,
        rock_aliases=rock_aliases,
        force_rock=args.force_rock_label,
    )

    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        text = f"{det['label']} {det['score'] * 100.0:.1f}%"
        draw_box(vis, (x1, y1, x2, y2), text)

    out_img = out_dir / f"{image_path.stem}_det{image_path.suffix}"
    cv2.imwrite(str(out_img), vis)

    if args.save_json:
        out_json = out_dir / f"{image_path.stem}_det.json"
        out_json.write_text(json.dumps({"image": image_path.name, "detections": detections}, indent=2))

    if args.show:
        cv2.imshow("AnyObjectDetect", vis)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            return -1

    print(f"[DONE] {image_path.name}: kept {len(detections)} objects")
    return len(detections)


def process_video(
    video_path: Path,
    model,
    args,
    ignore_labels: set,
    rock_aliases: set,
    out_dir: Path,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    out_path = out_dir / f"{video_path.stem}_det.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_idx = 0
    total_kept = 0
    all_meta = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = run_inference_on_frame(
            model,
            frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            ignore_labels=ignore_labels,
            min_bottom_frac=args.min_bottom_frac,
            min_box_area_px=args.min_box_area_px,
            rock_aliases=rock_aliases,
            force_rock=args.force_rock_label,
        )

        vis = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
            text = f"{det['label']} {det['score'] * 100.0:.1f}%"
            draw_box(vis, (x1, y1, x2, y2), text)

        cv2.putText(
            vis,
            f"frame={frame_idx} kept={len(detections)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
        )

        writer.write(vis)

        if args.save_json:
            all_meta.append({"frame": frame_idx, "detections": detections})

        if args.show:
            cv2.imshow("AnyObjectDetect", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        frame_idx += 1
        total_kept += len(detections)

    cap.release()
    writer.release()

    if args.save_json:
        out_json = out_dir / f"{video_path.stem}_det.json"
        out_json.write_text(json.dumps({"video": video_path.name, "frames": all_meta}, indent=2))

    print(f"[DONE] {video_path.name}: frames={frame_idx} kept_total={total_kept}")
    print(f"  Saved video: {out_path}")
    return total_kept


def iter_sources(src: Path):
    if src.is_file():
        return [src]
    if src.is_dir():
        items = [p for p in sorted(src.rglob("*")) if p.is_file()]
        return items
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect many objects and ignore ground labels")
    parser.add_argument("--source", required=True, help="Image/video file or directory")
    parser.add_argument("--model", default="yolov8s-worldv2.pt", help="YOLO/YOLO-World model path")
    parser.add_argument(
        "--prompts",
        default="rock,stone,boulder,pole,marker,person,robot,cone,box,crate,cable,barrel",
        help="Comma-separated open-vocabulary prompts (YOLO-World)",
    )
    parser.add_argument("--prompts-file", default="DataTraining/object_prompts.txt", help="Optional prompt list file")
    parser.add_argument(
        "--ignore-labels",
        default="ground,floor,sand,dirt,soil,terrain,road,path,wall,sky",
        help="Comma-separated labels to ignore",
    )
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="cpu", help="cpu or cuda id (e.g., 0)")
    parser.add_argument("--min-bottom-frac", type=float, default=0.35, help="Keep detections with bbox bottom >= this fraction of image height")
    parser.add_argument("--min-box-area-px", type=int, default=225, help="Minimum bbox area in pixels")
    parser.add_argument("--save-dir", default="DataTraining/runs/any_object_detect", help="Output folder")
    parser.add_argument("--save-json", action="store_true", help="Save detections as JSON")
    parser.add_argument("--show", action="store_true", help="Show detection window")
    parser.add_argument("--force-rock-label", action="store_true", help="Map rock-like labels to 'rock'")
    parser.add_argument(
        "--rock-aliases",
        default="rock,stone,boulder,cobble,gravel,barrel",
        help="Comma-separated labels to remap to 'rock' when --force-rock-label is enabled",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"[ERROR] Source not found: {source}")
        return 1

    prompt_items = read_lines(args.prompts_file)
    if not prompt_items:
        prompt_items = parse_items(args.prompts)
    if not prompt_items:
        print("[ERROR] No prompts provided")
        return 1

    ignore_items = parse_items(args.ignore_labels)
    ignore_set = build_set(ignore_items)

    rock_aliases = build_set(parse_items(args.rock_aliases))

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    # YOLO-World supports set_classes for open-vocabulary prompts.
    try:
        model.set_classes(prompt_items)
        print(f"[INFO] Open-vocabulary prompts set: {prompt_items}")
    except Exception:
        print("[INFO] Model does not support set_classes; using native model classes")

    items = iter_sources(source)
    if not items:
        print("[ERROR] No source files found")
        return 1

    total_kept = 0
    for item in items:
        ext = item.suffix.lower()
        if ext in IMAGE_EXTS:
            out = process_image(item, model, args, ignore_set, rock_aliases, out_dir)
            if out < 0:
                break
            total_kept += max(0, out)
        elif ext in VIDEO_EXTS:
            total_kept += process_video(item, model, args, ignore_set, rock_aliases, out_dir)

    if args.show:
        cv2.destroyAllWindows()

    print(f"[DONE] Completed. Total kept detections: {total_kept}")
    print(f"[DONE] Outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
