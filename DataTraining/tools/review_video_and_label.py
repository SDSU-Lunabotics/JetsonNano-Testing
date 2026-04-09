#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

try:
    import cv2
except Exception:
    print("[ERROR] Missing dependency: cv2")
    print("Install dependencies first:")
    print("  pip install -r DataTraining/requirements.txt")
    raise SystemExit(1)


def load_classes(path: Path) -> List[str]:
    classes = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not classes:
        raise ValueError(f"No classes in {path}")
    return classes


def choose_class_id(class_names: List[str], default_id: int = 0) -> int:
    print("\nChoose class id:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    raw = input(f"class id [{default_id}]: ").strip()
    if raw == "":
        return default_id
    try:
        cid = int(raw)
        if 0 <= cid < len(class_names):
            return cid
    except Exception:
        pass
    print("Invalid class id; using default")
    return default_id


def save_frame_and_label(frame, frame_idx: int, boxes, out_images: Path, out_labels: Path, stem: str) -> None:
    h, w = frame.shape[:2]
    image_name = f"{stem}_f{frame_idx:06d}.jpg"
    label_name = f"{stem}_f{frame_idx:06d}.txt"
    image_path = out_images / image_name
    label_path = out_labels / label_name

    cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    lines = []
    for cls_id, x1, y1, x2, y2 in boxes:
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        xc = x1 + bw / 2.0
        yc = y1 + bh / 2.0
        lines.append(f"{cls_id} {xc / w:.6f} {yc / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"[SAVE] {image_path.name} + {label_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Review a video and label selected frames into YOLO format")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--classes", default="DataTraining/classes.txt", help="Class names file")
    parser.add_argument("--out-images", default="DataTraining/data/raw_images", help="Output images folder")
    parser.add_argument("--out-labels", default="DataTraining/data/raw_labels", help="Output labels folder")
    parser.add_argument("--fps", type=float, default=15.0, help="Playback fps")
    args = parser.parse_args()

    video_path = Path(args.video)
    class_file = Path(args.classes)
    out_images = Path(args.out_images)
    out_labels = Path(args.out_labels)

    if not video_path.exists():
        print(f"[ERROR] Missing video: {video_path}")
        return 1
    if not class_file.exists():
        print(f"[ERROR] Missing classes file: {class_file}")
        return 1

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    class_names = load_classes(class_file)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 1

    delay_ms = max(1, int(1000.0 / max(1.0, args.fps)))
    paused = False
    last_class_id = 0
    frame_idx = -1
    frame = None

    print("Keys:")
    print("  space: pause/resume")
    print("  a: annotate current frame (draw 1+ boxes, then save)")
    print("  n: step 1 frame forward (while paused)")
    print("  q: quit")

    cv2.namedWindow("VideoLabel", cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ok, new_frame = cap.read()
            if not ok:
                print("[DONE] End of video")
                break
            frame_idx += 1
            frame = new_frame

        if frame is None:
            break

        overlay = frame.copy()
        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(
            overlay,
            f"{status} frame={frame_idx}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            "space pause/resume | a annotate+save | n step (paused) | q quit",
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )
        cv2.imshow("VideoLabel", overlay)

        key = cv2.waitKey(delay_ms if not paused else 50) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("n") and paused:
            ok, new_frame = cap.read()
            if ok:
                frame_idx += 1
                frame = new_frame
            else:
                print("[DONE] End of video")
                break
            continue
        if key == ord("a"):
            paused = True
            boxes = []
            while True:
                roi = cv2.selectROI("VideoLabel", frame, fromCenter=False, showCrosshair=True)
                x, y, rw, rh = roi
                if rw <= 1 or rh <= 1:
                    break
                cls_id = choose_class_id(class_names, last_class_id)
                last_class_id = cls_id
                boxes.append((cls_id, float(x), float(y), float(x + rw), float(y + rh)))
                again = input("Add another box on this frame? [y/N]: ").strip().lower()
                if again not in {"y", "yes"}:
                    break

            if boxes:
                save_frame_and_label(frame, frame_idx, boxes, out_images, out_labels, video_path.stem)
            else:
                print("[INFO] No boxes saved for this frame")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
