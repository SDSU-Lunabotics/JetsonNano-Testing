#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

try:
    import cv2
except Exception:
    print("[ERROR] Missing dependency: cv2")
    print("Install dependencies first:")
    print("  pip install -r DataTraining/requirements.txt")
    raise SystemExit(1)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm"}


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def copy_image(src: Path, output_dir: Path) -> int:
    dst = unique_path(output_dir / src.name)
    shutil.copy2(src, dst)
    return 1


def extract_video(video_path: Path, output_dir: Path, every_n_frames: int, jpg_quality: int, max_frames: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return 0

    written = 0
    frame_idx = 0
    base = video_path.stem

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every_n_frames == 0:
            out_name = f"{base}_f{frame_idx:06d}.jpg"
            out_path = unique_path(output_dir / out_name)
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
            written += 1
            if max_frames > 0 and written >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy images and extract video frames for YOLO labeling")
    parser.add_argument("--input-dir", default="DataTraining/data/raw_media", help="Folder with source images/videos")
    parser.add_argument("--output-dir", default="DataTraining/data/raw_images", help="Folder for extracted images")
    parser.add_argument("--video-every-n-frames", type=int, default=10, help="Take one frame every N video frames")
    parser.add_argument("--video-max-frames", type=int, default=0, help="Max extracted frames per video (0 = unlimited)")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality for extracted frames")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] Input folder not found: {input_dir}")
        return 1

    files = sorted([p for p in input_dir.rglob("*") if p.is_file()])
    if not files:
        print(f"[INFO] No files found in {input_dir}")
        return 0

    image_count = 0
    frame_count = 0

    for src in files:
        ext = src.suffix.lower()
        if ext in IMAGE_EXTS:
            image_count += copy_image(src, output_dir)
        elif ext in VIDEO_EXTS:
            extracted = extract_video(
                src,
                output_dir,
                every_n_frames=max(1, args.video_every_n_frames),
                jpg_quality=max(1, min(100, args.jpg_quality)),
                max_frames=max(0, args.video_max_frames),
            )
            frame_count += extracted
            print(f"[INFO] {src.name}: extracted {extracted} frames")

    print("[DONE] Data preparation complete")
    print(f"  Copied images: {image_count}")
    print(f"  Extracted frames: {frame_count}")
    print(f"  Output folder: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
