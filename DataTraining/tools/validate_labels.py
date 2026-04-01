#!/usr/bin/env python3
"""
Validate and repair YOLO format labels in the raw_labels directory.
Checks for format errors and helps identify issues before training.
"""
import argparse
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_classes(path: Path) -> List[str]:
    """Load class names from file."""
    lines = [line.strip() for line in path.read_text().splitlines()]
    classes = [line for line in lines if line and not line.startswith("#")]
    if not classes:
        raise ValueError(f"No classes found in {path}")
    return classes


def validate_yolo_line(line: str, num_classes: int, line_num: int) -> Tuple[bool, str]:
    """
    Validate a single YOLO format line.
    Format: class_id x_center y_center width height (all normalized 0-1)
    Returns: (is_valid, error_message)
    """
    line = line.strip()
    if not line:
        return True, ""  # Empty lines are OK

    parts = line.split()
    if len(parts) != 5:
        return False, f"Line {line_num}: Expected 5 fields, got {len(parts)}: {line}"

    try:
        cls_id = int(parts[0])
        if cls_id < 0 or cls_id >= num_classes:
            return False, f"Line {line_num}: Invalid class ID {cls_id} (valid: 0-{num_classes-1})"

        xc, yc, w, h = [float(x) for x in parts[1:]]

        # Check bounds (should be 0-1 for normalized YOLO format)
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return False, f"Line {line_num}: Values out of bounds [0-1]: xc={xc} yc={yc} w={w} h={h}"

        return True, ""
    except ValueError as e:
        return False, f"Line {line_num}: Parse error: {e}"


def validate_label_file(label_path: Path, num_classes: int) -> Tuple[bool, List[str]]:
    """
    Validate a label file.
    Returns: (is_valid, error_messages)
    """
    if not label_path.exists():
        return True, []  # File doesn't exist = valid (no labels yet)

    errors = []
    lines = label_path.read_text().splitlines()

    for line_num, line in enumerate(lines, start=1):
        is_valid, error = validate_yolo_line(line, num_classes, line_num)
        if not is_valid:
            errors.append(error)

    return len(errors) == 0, errors


def get_label_stats(label_path: Path, class_names: List[str]) -> dict:
    """Get statistics about a label file."""
    if not label_path.exists():
        return {"total_boxes": 0, "by_class": {}}

    stats = {"total_boxes": 0, "by_class": {c: 0 for c in class_names}}

    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            cls_id = int(line.split()[0])
            if 0 <= cls_id < len(class_names):
                class_name = class_names[cls_id]
                stats["by_class"][class_name] = stats["by_class"].get(class_name, 0) + 1
                stats["total_boxes"] += 1
        except Exception:
            pass

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate YOLO format labels")
    parser.add_argument("--images-dir", default="DataTraining/data/raw_images", help="Images directory")
    parser.add_argument("--labels-dir", default="DataTraining/data/raw_labels", help="Labels directory")
    parser.add_argument("--classes", default="DataTraining/classes.txt", help="Classes file")
    parser.add_argument("--show-stats", action="store_true", help="Show label statistics")
    parser.add_argument("--fix-missing", action="store_true", help="Create empty label files for images without labels")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    class_file = Path(args.classes)

    if not images_dir.exists():
        print(f"[ERROR] Images dir not found: {images_dir}")
        return 1
    if not class_file.exists():
        print(f"[ERROR] Classes file not found: {class_file}")
        return 1

    class_names = load_classes(class_file)
    num_classes = len(class_names)

    print(f"[INFO] Found {num_classes} classes: {class_names}")

    # Find all images
    image_paths = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    print(f"[INFO] Found {len(image_paths)} images")

    # Validate each label
    errors_found = False
    valid_count = 0
    missing_count = 0
    total_boxes = 0

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            missing_count += 1
            status = "[MISSING]"
            if args.fix_missing:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text("")
                status += " (created)"
        else:
            is_valid, errors = validate_label_file(label_path, num_classes)
            if is_valid:
                valid_count += 1
                status = "[OK]"
            else:
                errors_found = True
                status = "[ERROR]"
                for error in errors:
                    print(f"  {error}")

            stats = get_label_stats(label_path, class_names)
            total_boxes += stats["total_boxes"]
            boxes_str = f"{stats['total_boxes']} boxes"
            if args.show_stats and stats["total_boxes"] > 0:
                class_breakdown = ", ".join([f"{c}:{n}" for c, n in stats["by_class"].items() if n > 0])
                boxes_str += f" ({class_breakdown})"
            status += f" {boxes_str}"

        print(f"{status:30} {image_path.name}")

    print(f"\n[SUMMARY]")
    print(f"  Valid labels:    {valid_count}")
    print(f"  Missing labels:  {missing_count}")
    print(f"  Total boxes:     {total_boxes}")
    print(f"  Classes:         {num_classes}")

    if errors_found:
        print(f"\n[WARNING] Some labels have errors. Please review and fix manually.")
        return 1
    else:
        print(f"\n[OK] All labels are valid!")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
