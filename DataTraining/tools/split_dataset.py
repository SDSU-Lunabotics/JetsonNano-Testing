#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_classes(path: Path) -> List[str]:
    classes = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not classes:
        raise ValueError(f"No classes in {path}")
    return classes


def ensure_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def parse_ratios(ratios: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in ratios.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected exactly 3 ratios, e.g. 0.8,0.15,0.05")
    s = sum(parts)
    if s <= 0:
        raise ValueError("Ratios sum must be > 0")
    return tuple(x / s for x in parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create YOLO train/val/test splits")
    parser.add_argument("--images-dir", default="DataTraining/data/raw_images")
    parser.add_argument("--labels-dir", default="DataTraining/data/raw_labels")
    parser.add_argument("--output-dir", default="DataTraining/data/dataset")
    parser.add_argument("--classes", default="DataTraining/classes.txt")
    parser.add_argument("--ratios", default="0.8,0.15,0.05", help="train,val,test ratios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-empty", action="store_true", help="Include images without labels")
    parser.add_argument("--dataset-yaml", default="dataset.yaml", help="YAML filename under output-dir")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    classes_file = Path(args.classes)

    if not images_dir.exists():
        print(f"[ERROR] Missing images dir: {images_dir}")
        return 1
    if not labels_dir.exists():
        print(f"[ERROR] Missing labels dir: {labels_dir}")
        return 1
    if not classes_file.exists():
        print(f"[ERROR] Missing classes file: {classes_file}")
        return 1

    classes = load_classes(classes_file)
    train_r, val_r, test_r = parse_ratios(args.ratios)

    all_images = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    samples = []
    for img in all_images:
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists() or args.include_empty:
            samples.append((img, lbl))

    if not samples:
        print("[ERROR] No usable samples found. Add labels or use --include-empty")
        return 1

    random.seed(args.seed)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    n_test = n - n_train - n_val

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    ensure_dirs(output_dir)

    copied = {"train": 0, "val": 0, "test": 0}
    for split, items in splits.items():
        for img, lbl in items:
            out_img = output_dir / "images" / split / img.name
            out_lbl = output_dir / "labels" / split / f"{img.stem}.txt"
            shutil.copy2(img, out_img)
            if lbl.exists():
                shutil.copy2(lbl, out_lbl)
            else:
                out_lbl.write_text("")
            copied[split] += 1

    yaml_path = output_dir / args.dataset_yaml
    names_yaml = "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])
    yaml_text = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n{names_yaml}\n"
    )
    yaml_path.write_text(yaml_text)

    print("[DONE] Dataset split complete")
    print(f"  Total samples: {n}")
    print(f"  Train: {copied['train']}")
    print(f"  Val:   {copied['val']}")
    print(f"  Test:  {copied['test']}")
    print(f"  Dataset YAML: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
