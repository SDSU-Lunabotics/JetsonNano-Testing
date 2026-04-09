#!/usr/bin/env python3
"""
Clean label files: remove confidence scores for standard 5-column YOLO format.
Converts: cls x y w h conf  →  cls x y w h
"""
from pathlib import Path

def clean_labels(labels_dir: Path) -> None:
    """Remove confidence column from all label files."""
    cleaned = 0
    
    for label_file in labels_dir.glob("*.txt"):
        content = label_file.read_text()
        if not content.strip():
            continue
        
        lines = content.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) >= 5:
                # Keep only first 5 columns (class_id x y w h)
                cleaned_line = ' '.join(parts[:5])
                cleaned_lines.append(cleaned_line)
        
        # Write back
        new_content = '\n'.join(cleaned_lines)
        if cleaned_lines:
            new_content += '\n'
        label_file.write_text(new_content)
        cleaned += 1
    
    print(f"[DONE] Cleaned {cleaned} label files")
    print(f"  Removed confidence column (6th column)")
    print(f"  Format now: class_id x y w h (5 columns)")

if __name__ == "__main__":
    raw_labels = Path("DataTraining/data/raw_labels")
    if raw_labels.exists():
        print(f"[INFO] Cleaning {raw_labels}...")
        clean_labels(raw_labels)
    
    dataset_labels = Path("DataTraining/data/dataset/labels")
    if dataset_labels.exists():
        print(f"[INFO] Cleaning {dataset_labels}...")
        for d in [dataset_labels / "train", dataset_labels / "val", dataset_labels / "test"]:
            if d.exists():
                clean_labels(d)
