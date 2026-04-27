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
    parser = argparse.ArgumentParser(description="Export trained model for Jetson deployment")
    parser.add_argument("--model", default="DataTraining/runs/pole_rock_train/weights/best.pt")
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "torchscript", "openvino"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 where supported")
    parser.add_argument("--device", default=None, help="Set for engine export on Jetson, e.g. 0")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args()

    if args.format == "engine":
        print("[INFO] TensorRT engine export is best run on the target Jetson device.")

    model = YOLO(args.model)
    out = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )

    print("[DONE] Export complete")
    print(f"  Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
