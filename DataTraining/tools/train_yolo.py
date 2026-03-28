#!/usr/bin/env python3
import argparse

try:
    from ultralytics import YOLO
except Exception:
    print("[ERROR] Missing dependency: ultralytics")
    print("Install dependencies first:")
    print("  pip install -r DataTraining/requirements.txt")
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO model on prepared dataset")
    parser.add_argument("--data", default="DataTraining/data/dataset/dataset.yaml", help="Path to dataset YAML")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", help="cpu, 0, 0,1, ...")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="DataTraining/runs")
    parser.add_argument("--name", default="pole_rock_train")
    args = parser.parse_args()

    model = YOLO(args.model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )

    print("[DONE] Training complete")
    print(f"  Save dir: {result.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
