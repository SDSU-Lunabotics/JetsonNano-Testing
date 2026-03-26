#!/usr/bin/env python3
"""
Live perception testing lab for ZED camera.

Goals:
- Tune geometric obstacle detection live (trackbars).
- Visualize obstacle boxes from depth/plane geometry.
- Optionally overlay AI object boxes (YOLO via ultralytics).

This script is intentionally separate from ZEDAuto so you can test quickly
without affecting autonomous drive code.
"""

import argparse
import csv
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:
    print("OpenCV is required for this tool.")
    print(f"Error: {exc}")
    sys.exit(1)

try:
    import pyzed.sl as sl
except Exception as exc:
    print("Failed to import pyzed.sl. Install ZED SDK Python API on this machine.")
    print(f"Error: {exc}")
    sys.exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
ZEDAUTO_DIR = os.path.join(REPO_ROOT, "ZEDAuto")
if ZEDAUTO_DIR not in sys.path:
    sys.path.insert(0, ZEDAUTO_DIR)

import segmentation
import zed_utils


def _noop(_value: int) -> None:
    return


def _class_color(idx: int) -> Tuple[int, int, int]:
    palette = [
        (0, 255, 255),   # yellow
        (255, 0, 255),   # magenta
        (255, 180, 0),   # cyan-ish
        (0, 180, 255),   # orange-ish
        (180, 255, 0),   # lime
        (0, 120, 255),   # orange
        (255, 120, 0),   # blue-orange alt
        (140, 180, 255), # light orange
        (200, 200, 200), # gray
    ]
    return palette[idx % len(palette)]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _first_enum_attr(enum_obj: object, names: List[str]) -> Optional[object]:
    for n in names:
        if hasattr(enum_obj, n):
            return getattr(enum_obj, n)
    return None


def _clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_to_yolo(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + (bw / 2.0)
    cy = y1 + (bh / 2.0)
    return cx / float(w), cy / float(h), bw / float(w), bh / float(h)


def _world_to_grid(
    x: np.ndarray,
    z: np.ndarray,
    x_min: float,
    z_min: float,
    map_res_m: float,
    grid_h: int,
    grid_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    col = ((x - x_min) / map_res_m).astype(np.int32)
    row = (grid_h - 1 - ((z - z_min) / map_res_m)).astype(np.int32)
    inb = (row >= 0) & (row < grid_h) & (col >= 0) & (col < grid_w)
    return row, col, inb


def _render_semantic_map(
    sem_counts: np.ndarray,
    class_names: List[str],
    cam_row_col: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    # sem_counts: (C, H, W)
    cnum, h, w = sem_counts.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    sums = np.sum(sem_counts, axis=0)
    dom = np.argmax(sem_counts, axis=0)
    known = sums > 0.0

    for ci in range(cnum):
        mask = known & (dom == ci)
        if not np.any(mask):
            continue
        color = np.array(_class_color(ci), dtype=np.uint8)
        out[mask] = color

    if cam_row_col is not None:
        rr, cc = cam_row_col
        rr0 = max(0, rr - 2)
        rr1 = min(h, rr + 3)
        cc0 = max(0, cc - 2)
        cc1 = min(w, cc + 3)
        out[rr0:rr1, cc0:cc1] = (255, 255, 255)

    # Legend panel
    legend_h = max(20 * (len(class_names) + 1), 80)
    legend = np.zeros((legend_h, 250, 3), dtype=np.uint8)
    cv2.putText(legend, "Semantic Map", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    y = 38
    for i, name in enumerate(class_names):
        col = _class_color(i)
        cv2.rectangle(legend, (10, y - 10), (24, y + 2), col, -1)
        cv2.putText(
            legend,
            f"{i + 1}:{name}",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return np.hstack((out, legend))


class OptionalYoloDetector:
    def __init__(
        self,
        model_path: Optional[str],
        labels_path: Optional[str],
        imgsz: int,
        every_n: int,
        device: str,
    ) -> None:
        self.available = False
        self.model = None
        self.class_names = {}
        self.every_n = max(1, int(every_n))
        self.imgsz = int(imgsz)
        self.device = device
        self.last_boxes: List[Tuple[int, int, int, int, float, str]] = []
        self.last_frame_idx = -999999

        if not model_path:
            return
        if not os.path.exists(model_path):
            print(f"AI model file not found: {model_path}")
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            print("ultralytics not installed; AI detector disabled.")
            print(f"Install with: pip install ultralytics")
            print(f"Import error: {exc}")
            return

        try:
            self.model = YOLO(model_path)
            self.available = True
            if labels_path and os.path.exists(labels_path):
                with open(labels_path, "r", encoding="utf-8") as f:
                    labels = [line.strip() for line in f if line.strip()]
                self.class_names = {idx: name for idx, name in enumerate(labels)}
            print(f"AI detector loaded: {model_path}")
        except Exception as exc:
            print(f"Failed to load AI model: {exc}")
            self.available = False
            self.model = None

    def detect(
        self,
        frame_bgr: np.ndarray,
        conf_thresh: float,
        iou_thresh: float,
        frame_idx: int,
    ) -> List[Tuple[int, int, int, int, float, str]]:
        if not self.available or self.model is None:
            return []
        if (frame_idx - self.last_frame_idx) < self.every_n:
            return self.last_boxes

        self.last_frame_idx = frame_idx
        h, w = frame_bgr.shape[:2]
        out: List[Tuple[int, int, int, int, float, str]] = []
        try:
            res = self.model.predict(
                source=frame_bgr,
                conf=float(conf_thresh),
                iou=float(iou_thresh),
                imgsz=self.imgsz,
                device=self.device if self.device else None,
                verbose=False,
            )
            if not res:
                self.last_boxes = []
                return []
            r0 = res[0]
            boxes = getattr(r0, "boxes", None)
            if boxes is None:
                self.last_boxes = []
                return []
            names_map = getattr(r0, "names", {}) or {}
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))
                conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
                cls_idx = int(b.cls[0].item()) if hasattr(b, "cls") else -1
                label = self.class_names.get(cls_idx)
                if label is None:
                    label = names_map.get(cls_idx, f"class_{cls_idx}")
                out.append((x1, y1, x2, y2, conf, str(label)))
        except Exception as exc:
            print(f"AI detect error: {exc}")
            out = []

        self.last_boxes = out
        return out


class OptionalZedSdkDetector:
    def __init__(
        self,
        zed_cam: "sl.Camera",
        enabled: bool,
        confidence: int,
        every_n: int,
        use_tracking: bool,
    ) -> None:
        self.available = False
        self.zed = zed_cam
        self.objects = None
        self.runtime = None
        self.every_n = max(1, int(every_n))
        self.last_boxes: List[Tuple[int, int, int, int, float, str]] = []
        self.last_frame_idx = -999999
        self.confidence = int(max(1, min(99, int(confidence))))

        if not enabled:
            return
        if not hasattr(sl, "ObjectDetectionParameters"):
            print("ZED SDK ObjectDetectionParameters not found; ZED detector disabled.")
            return

        try:
            params = sl.ObjectDetectionParameters()
            if hasattr(params, "enable_tracking"):
                params.enable_tracking = bool(use_tracking)
            model_enum = getattr(sl, "OBJECT_DETECTION_MODEL", None)
            if model_enum is not None and hasattr(params, "detection_model"):
                model_val = _first_enum_attr(
                    model_enum,
                    [
                        "MULTI_CLASS_BOX_MEDIUM",
                        "MULTI_CLASS_BOX_FAST",
                        "MULTI_CLASS_BOX_ACCURATE",
                        "MULTI_CLASS_BOX",
                    ],
                )
                if model_val is not None:
                    params.detection_model = model_val

            err = self.zed.enable_object_detection(params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"ZED built-in object detection enable failed: {err}")
                return

            self.objects = sl.Objects()
            self.runtime = sl.ObjectDetectionRuntimeParameters()
            if hasattr(self.runtime, "detection_confidence_threshold"):
                self.runtime.detection_confidence_threshold = int(self.confidence)
            self.available = True
            print("ZED built-in object detection enabled.")
        except Exception as exc:
            print(f"Failed to initialize ZED built-in object detection: {exc}")
            self.available = False

    def detect(self, frame_idx: int, conf_percent: Optional[int] = None) -> List[Tuple[int, int, int, int, float, str]]:
        if not self.available or self.objects is None or self.runtime is None:
            return []
        if (frame_idx - self.last_frame_idx) < self.every_n:
            return self.last_boxes
        self.last_frame_idx = frame_idx

        if conf_percent is not None and hasattr(self.runtime, "detection_confidence_threshold"):
            self.runtime.detection_confidence_threshold = int(max(1, min(99, int(conf_percent))))

        out: List[Tuple[int, int, int, int, float, str]] = []
        try:
            err = self.zed.retrieve_objects(self.objects, self.runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                self.last_boxes = []
                return []
            obj_list = getattr(self.objects, "object_list", [])
            for obj in obj_list:
                bb = getattr(obj, "bounding_box_2d", None)
                if bb is None:
                    continue
                pts = np.array(bb, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] == 0:
                    continue
                x1 = int(np.floor(np.min(pts[:, 0])))
                y1 = int(np.floor(np.min(pts[:, 1])))
                x2 = int(np.ceil(np.max(pts[:, 0])))
                y2 = int(np.ceil(np.max(pts[:, 1])))
                if x2 <= x1 or y2 <= y1:
                    continue

                conf = float(getattr(obj, "confidence", 0.0)) / 100.0
                raw_label = getattr(obj, "label", None)
                label = str(raw_label) if raw_label is not None else "zed_object"
                sublabel = getattr(obj, "sublabel", "")
                if isinstance(sublabel, str) and sublabel:
                    label = sublabel
                out.append((x1, y1, x2, y2, conf, label))
        except Exception as exc:
            print(f"ZED built-in detect error: {exc}")
            out = []

        self.last_boxes = out
        return out

    def close(self) -> None:
        if not self.available:
            return
        try:
            self.zed.disable_object_detection()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live ZED perception testing tool")
    p.add_argument("--tracking", action="store_true", help="Enable ZED positional tracking")
    p.add_argument("--floor-update-sec", type=float, default=0.5, help="Seconds between floor-plane updates")
    p.add_argument("--floor-min-normal-y", type=float, default=0.5, help="Reject floor plane if |normal.y| is lower")
    p.add_argument("--stride", type=int, default=8, help="Point-cloud downsample stride for geometry masks")
    p.add_argument("--obstacle-thresh-m", type=float, default=0.05, help="Obstacle height above floor plane (m)")
    p.add_argument("--hole-thresh-m", type=float, default=0.10, help="Hole depth below floor plane (m)")
    p.add_argument("--max-above-ground-m", type=float, default=1.22, help="Ignore points above this floor-relative height (m)")
    p.add_argument("--max-forward-m", type=float, default=6.0, help="Ignore points farther than this forward distance (m)")
    p.add_argument("--min-box-area-px", type=int, default=1200, help="Minimum geometric obstacle box area in pixels")
    p.add_argument("--ai-model", default="", help="Optional YOLO model path (.pt/.onnx) for AI boxes")
    p.add_argument("--ai-labels", default="", help="Optional labels txt path (one class per line)")
    p.add_argument("--ai-conf", type=float, default=0.40, help="AI confidence threshold")
    p.add_argument("--ai-iou", type=float, default=0.45, help="AI IoU threshold")
    p.add_argument("--ai-every", type=int, default=2, help="Run AI every N frames")
    p.add_argument("--ai-imgsz", type=int, default=640, help="AI inference image size")
    p.add_argument("--ai-device", default="", help="AI device override (e.g. cuda:0, cpu)")
    p.add_argument(
        "--detector-mode",
        default="zed",
        choices=["none", "yolo", "zed", "both"],
        help="Object detector source",
    )
    p.add_argument("--zed-od-confidence", type=int, default=40, help="ZED built-in OD confidence threshold (1..99)")
    p.add_argument("--zed-od-every", type=int, default=1, help="Run ZED built-in OD every N frames")
    p.add_argument("--zed-od-tracking", action="store_true", help="Enable ZED OD tracking (requires tracking)")
    p.add_argument(
        "--classes",
        default="rock,wall,person,cable,cone,other",
        help="Comma-separated class names for annotation hotkeys 1..9",
    )
    p.add_argument(
        "--dataset-dir",
        default=os.path.join(SCRIPT_DIR, "dataset"),
        help="Where labeled images/labels/metadata are saved",
    )
    p.add_argument("--annotation-mode", action="store_true", help="Start with annotation mode enabled")
    p.add_argument("--semantic-map", action="store_true", help="Show semantic top-down map from labeled boxes")
    p.add_argument("--map-width-m", type=float, default=20.0, help="Semantic map width in meters (X)")
    p.add_argument("--map-height-m", type=float, default=20.0, help="Semantic map height in meters (Z)")
    p.add_argument("--map-res-m", type=float, default=0.05, help="Semantic map resolution (m/cell)")
    p.add_argument("--map-center", action="store_true", help="Center semantic map around Z=0")
    p.add_argument("--semantic-point-stride", type=int, default=4, help="Pixel stride when projecting labeled boxes to map")
    p.add_argument("--semantic-decay", type=float, default=1.0, help="Semantic map decay per frame (1.0=none)")
    p.add_argument("--ground-band-m", type=float, default=0.10, help="Floor band for ground class projection (m)")
    return p.parse_args()


def create_controls(args: argparse.Namespace) -> None:
    cv2.namedWindow("Perception Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Perception Controls", 560, 420)
    cv2.createTrackbar("Obstacle cm", "Perception Controls", int(max(1.0, args.obstacle_thresh_m * 100.0)), 200, _noop)
    cv2.createTrackbar("Hole cm", "Perception Controls", int(max(1.0, args.hole_thresh_m * 100.0)), 200, _noop)
    cv2.createTrackbar("MaxAbove cm", "Perception Controls", int(max(0.0, args.max_above_ground_m * 100.0)), 500, _noop)
    cv2.createTrackbar("MaxFwd dm", "Perception Controls", int(max(0.0, args.max_forward_m * 10.0)), 300, _noop)
    cv2.createTrackbar("Stride", "Perception Controls", int(max(2, args.stride)), 20, _noop)
    cv2.createTrackbar("MinBox x100", "Perception Controls", int(max(1, args.min_box_area_px // 100)), 500, _noop)
    cv2.createTrackbar("AI Conf %", "Perception Controls", int(max(1.0, min(99.0, args.ai_conf * 100.0))), 100, _noop)
    cv2.createTrackbar("AI IoU %", "Perception Controls", int(max(1.0, min(99.0, args.ai_iou * 100.0))), 100, _noop)
    cv2.createTrackbar("ShowGeom", "Perception Controls", 1, 1, _noop)
    cv2.createTrackbar("ShowBoxes", "Perception Controls", 1, 1, _noop)
    ai_default_on = 1 if (args.ai_model or args.detector_mode in ("zed", "both")) else 0
    cv2.createTrackbar("ShowAI", "Perception Controls", ai_default_on, 1, _noop)


def read_controls() -> dict:
    stride = max(2, cv2.getTrackbarPos("Stride", "Perception Controls"))
    obstacle_thresh_m = max(0.01, cv2.getTrackbarPos("Obstacle cm", "Perception Controls") / 100.0)
    hole_thresh_m = max(0.01, cv2.getTrackbarPos("Hole cm", "Perception Controls") / 100.0)
    max_above_ground_m = cv2.getTrackbarPos("MaxAbove cm", "Perception Controls") / 100.0
    max_forward_m = cv2.getTrackbarPos("MaxFwd dm", "Perception Controls") / 10.0
    min_box_area_px = max(100, cv2.getTrackbarPos("MinBox x100", "Perception Controls") * 100)
    ai_conf = max(0.01, cv2.getTrackbarPos("AI Conf %", "Perception Controls") / 100.0)
    ai_iou = max(0.01, cv2.getTrackbarPos("AI IoU %", "Perception Controls") / 100.0)
    show_geom = cv2.getTrackbarPos("ShowGeom", "Perception Controls") == 1
    show_boxes = cv2.getTrackbarPos("ShowBoxes", "Perception Controls") == 1
    show_ai = cv2.getTrackbarPos("ShowAI", "Perception Controls") == 1
    return {
        "stride": stride,
        "obstacle_thresh_m": obstacle_thresh_m,
        "hole_thresh_m": hole_thresh_m,
        "max_above_ground_m": max_above_ground_m,
        "max_forward_m": max_forward_m,
        "min_box_area_px": min_box_area_px,
        "ai_conf": ai_conf,
        "ai_iou": ai_iou,
        "show_geom": show_geom,
        "show_boxes": show_boxes,
        "show_ai": show_ai,
    }


def normalize_bgr(img: np.ndarray) -> Optional[np.ndarray]:
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.shape[2] == 3:
            return img
        if img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img[:, :, :3]
    return None


def main() -> int:
    args = parse_args()
    class_names = [x.strip() for x in str(args.classes).split(",") if x.strip()]
    if not class_names:
        class_names = ["obstacle"]
    if len(class_names) > 9:
        class_names = class_names[:9]

    images_dir = os.path.join(args.dataset_dir, "images")
    labels_dir = os.path.join(args.dataset_dir, "labels")
    _ensure_dir(images_dir)
    _ensure_dir(labels_dir)
    meta_csv_path = os.path.join(args.dataset_dir, "annotations.csv")
    if not os.path.exists(meta_csv_path):
        with open(meta_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "image",
                    "label",
                    "class_id",
                    "class_name",
                    "source",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ]
            )

    print("Starting Perception Lab...")
    print("Keys: q=quit, p=pause, r=force floor update, s=snapshot, l=annotation mode")
    print("Annotation: drag/click box, then press 1..9 for class")

    zed = zed_utils.open_zed_camera(sl)
    runtime = sl.RuntimeParameters()
    point_cloud = sl.Mat()
    image_left = sl.Mat()
    ground_plane = sl.Plane()
    tracking_reset = sl.Transform()
    pose = sl.Pose()
    pose_warned = False

    if args.tracking:
        zed_utils.enable_tracking(zed, sl)

    detector_mode = str(args.detector_mode).lower()
    use_yolo = detector_mode in ("yolo", "both")
    use_zed = detector_mode in ("zed", "both")

    yolo = OptionalYoloDetector(
        model_path=args.ai_model if use_yolo else "",
        labels_path=args.ai_labels,
        imgsz=args.ai_imgsz,
        every_n=args.ai_every,
        device=args.ai_device,
    )
    if use_yolo and (not yolo.available) and args.ai_model:
        print("YOLO model configured but unavailable; continuing.")

    zed_detector = OptionalZedSdkDetector(
        zed_cam=zed,
        enabled=use_zed,
        confidence=int(args.zed_od_confidence),
        every_n=int(args.zed_od_every),
        use_tracking=bool(args.zed_od_tracking and args.tracking),
    )

    create_controls(args)
    cv2.namedWindow("Perception Lab", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Obstacle Mask", cv2.WINDOW_NORMAL)
    if args.semantic_map:
        cv2.namedWindow("Semantic Map (XZ)", cv2.WINDOW_NORMAL)

    has_plane = False
    a, b, c, d = 0.0, 1.0, 0.0, 0.0
    last_floor_update = 0.0
    paused = False
    frame_idx = 0
    fps_smooth = 0.0
    t_prev = time.time()
    annotation_mode = bool(args.annotation_mode)
    pending_box: Optional[Tuple[int, int, int, int]] = None
    pending_source = "manual"
    mouse_down = False
    drag_start: Optional[Tuple[int, int]] = None
    drag_curr: Optional[Tuple[int, int]] = None
    click_candidate_box: Optional[Tuple[int, int, int, int]] = None
    click_candidate_source = "manual"
    annotation_candidates: List[Tuple[int, int, int, int, str, str]] = []
    last_saved_msg = ""
    last_frame_shape = (720, 1280)
    brush_class_idx = 0
    map_mouse_down = False
    map_display_scale = 2
    # Semantic map state
    map_w = int(max(10, np.round(float(args.map_width_m) / float(args.map_res_m))))
    map_h = int(max(10, np.round(float(args.map_height_m) / float(args.map_res_m))))
    sem_counts = np.zeros((len(class_names), map_h, map_w), dtype=np.float32)
    map_x_min = -float(args.map_width_m) / 2.0
    map_z_min = -float(args.map_height_m) / 2.0 if args.map_center else 0.0

    def find_candidate_box(px: int, py: int) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
        for x1, y1, x2, y2, source, _label in annotation_candidates:
            if x1 <= px <= x2 and y1 <= py <= y2:
                return (x1, y1, x2, y2), source
        return None, "manual"

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal mouse_down, drag_start, drag_curr, pending_box, pending_source
        nonlocal click_candidate_box, click_candidate_source
        nonlocal last_frame_shape
        if not annotation_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_down = True
            drag_start = (x, y)
            drag_curr = (x, y)
            cand, src = find_candidate_box(x, y)
            click_candidate_box = cand
            click_candidate_source = src
        elif event == cv2.EVENT_MOUSEMOVE and mouse_down:
            drag_curr = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_down = False
            if drag_start is None:
                return
            ex, ey = x, y
            sx, sy = drag_start
            move = abs(ex - sx) + abs(ey - sy)
            if move < 8 and click_candidate_box is not None:
                pending_box = click_candidate_box
                pending_source = click_candidate_source
            else:
                hloc, wloc = int(last_frame_shape[0]), int(last_frame_shape[1])
                box = _clamp_box(min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey), wloc, hloc)
                if box is not None:
                    pending_box = box
                    pending_source = "manual"
            drag_start = None
            drag_curr = None
            click_candidate_box = None

    cv2.setMouseCallback("Perception Lab", on_mouse)

    def on_semantic_map_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal map_mouse_down
        if not args.semantic_map:
            return
        map_w_disp = map_w * map_display_scale
        map_h_disp = map_h * map_display_scale
        if x < 0 or y < 0 or x >= map_w_disp or y >= map_h_disp:
            return
        rr = int(y / map_display_scale)
        cc = int(x / map_display_scale)
        if rr < 0 or rr >= map_h or cc < 0 or cc >= map_w:
            return
        brush_r = 2

        def paint(add: bool) -> None:
            r0 = max(0, rr - brush_r)
            r1 = min(map_h, rr + brush_r + 1)
            c0 = max(0, cc - brush_r)
            c1 = min(map_w, cc + brush_r + 1)
            if add:
                sem_counts[:, r0:r1, c0:c1] *= 0.8
                sem_counts[brush_class_idx, r0:r1, c0:c1] += 8.0
            else:
                sem_counts[:, r0:r1, c0:c1] *= 0.0

        if event == cv2.EVENT_LBUTTONDOWN:
            map_mouse_down = True
            paint(True)
        elif event == cv2.EVENT_MOUSEMOVE and map_mouse_down:
            paint(True)
        elif event == cv2.EVENT_LBUTTONUP:
            map_mouse_down = False
            paint(True)
        elif event == cv2.EVENT_RBUTTONDOWN:
            paint(False)

    if args.semantic_map:
        cv2.setMouseCallback("Semantic Map (XZ)", on_semantic_map_mouse)

    def save_annotation(
        frame_bgr: np.ndarray,
        box: Tuple[int, int, int, int],
        class_id: int,
        source: str,
    ) -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1.0) * 1000)
        base = f"img_{stamp}_{ms:03d}_{frame_idx:06d}"
        img_name = f"{base}.jpg"
        lbl_name = f"{base}.txt"
        img_path = os.path.join(images_dir, img_name)
        lbl_path = os.path.join(labels_dir, lbl_name)

        hloc, wloc = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        cv2.imwrite(img_path, frame_bgr)
        cx, cy, bw, bh = _box_to_yolo(box, wloc, hloc)
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        with open(meta_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    time.time(),
                    img_name,
                    lbl_name,
                    class_id,
                    class_names[class_id],
                    source,
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )
        return img_name

    def project_box_to_semantic_map(
        cloud_xyz: np.ndarray,
        box: Tuple[int, int, int, int],
        class_id: int,
        R_world_cam: np.ndarray,
        t_world_cam: np.ndarray,
    ) -> int:
        if not args.semantic_map:
            return 0
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return 0
        step = max(1, int(args.semantic_point_stride))
        region = cloud_xyz[y1:y2:step, x1:x2:step, :3]
        if region.size == 0:
            return 0
        pts = region.reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if pts.size == 0:
            return 0
        if float(args.max_forward_m) > 0.0:
            pts = pts[(pts[:, 2] >= 0.0) & (pts[:, 2] <= float(args.max_forward_m))]
        if pts.size == 0:
            return 0

        class_name = class_names[class_id].lower()
        # Keep points by class type.
        denom = max(1e-6, float(np.sqrt(a * a + b * b + c * c)))
        dist = (a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / denom
        if ("ground" in class_name) or ("floor" in class_name):
            keep = np.abs(dist) <= max(0.02, float(args.ground_band_m))
        elif ("hole" in class_name) or ("pit" in class_name):
            keep = dist < -max(0.02, float(args.hole_thresh_m))
        else:
            keep = dist > max(0.02, float(args.obstacle_thresh_m) * 0.5)
            if float(args.max_above_ground_m) > 0.0:
                keep = keep & (dist <= float(args.max_above_ground_m))
        pts = pts[keep]
        if pts.size == 0:
            return 0

        pts_world = (R_world_cam @ pts.T).T + t_world_cam.reshape(1, 3)
        row, col, inb = _world_to_grid(
            pts_world[:, 0],
            pts_world[:, 2],
            map_x_min,
            map_z_min,
            float(args.map_res_m),
            map_h,
            map_w,
        )
        if not np.any(inb):
            return 0
        np.add.at(sem_counts[class_id], (row[inb], col[inb]), 1.0)
        return int(np.count_nonzero(inb))

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
                print(f"Paused: {paused}")
            if key == ord("r"):
                has_plane = False
                print("Floor plane reset requested.")
            if key == ord("l"):
                annotation_mode = not annotation_mode
                pending_box = None
                print(f"Annotation mode: {'ON' if annotation_mode else 'OFF'}")
            if key == ord("0"):
                pending_box = None

            selected_class_idx = -1
            if ord("1") <= key <= ord("9"):
                selected_class_idx = key - ord("1")
                if selected_class_idx < len(class_names):
                    brush_class_idx = selected_class_idx

            if paused:
                continue

            if 0.0 < float(args.semantic_decay) < 1.0:
                sem_counts *= float(args.semantic_decay)

            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            if args.tracking:
                R_world_cam, t_world_cam, pose_warned = zed_utils.get_world_transform(
                    zed, sl, pose, pose_warned
                )
            else:
                R_world_cam = np.eye(3, dtype=np.float32)
                t_world_cam = np.zeros(3, dtype=np.float32)

            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            frame_raw = image_left.get_data()
            frame = normalize_bgr(frame_raw)
            if frame is None:
                continue
            last_frame_shape = frame.shape[:2]

            controls = read_controls()
            now = time.time()
            if (not has_plane) or ((now - last_floor_update) >= max(0.05, float(args.floor_update_sec))):
                status = zed.find_floor_plane(ground_plane, tracking_reset)
                last_floor_update = now
                if status == sl.ERROR_CODE.SUCCESS:
                    a0, b0, c0, d0 = segmentation.plane_params(ground_plane)
                    a0, b0, c0, d0 = segmentation.normalize_plane(a0, b0, c0, d0)
                    if abs(float(b0)) >= float(args.floor_min_normal_y):
                        a, b, c, d = a0, b0, c0, d0
                        has_plane = True
                    else:
                        print(f"Rejected weak floor normal.y={b0:.3f}")
                else:
                    print(f"find_floor_plane failed: {status}")

            cloud = point_cloud.get_data()
            if cloud is None:
                continue

            stride = int(controls["stride"])
            xyz_small = cloud[::stride, ::stride, :3]
            valid = np.isfinite(xyz_small).all(axis=2)
            if controls["max_forward_m"] > 0.0:
                valid = valid & (xyz_small[:, :, 2] <= float(controls["max_forward_m"]))
                valid = valid & (xyz_small[:, :, 2] >= 0.0)

            dist_num = (a * xyz_small[:, :, 0] + b * xyz_small[:, :, 1] + c * xyz_small[:, :, 2] + d)
            denom = max(1e-6, float(np.sqrt(a * a + b * b + c * c)))
            dist_full = np.full_like(dist_num, np.nan, dtype=np.float32)
            if np.any(valid):
                dist_full[valid] = (dist_num[valid] / denom).astype(np.float32)

            obstacle_thresh = float(controls["obstacle_thresh_m"])
            hole_thresh = float(controls["hole_thresh_m"])
            max_above = float(controls["max_above_ground_m"])

            ground_small = (np.abs(dist_full) < obstacle_thresh) & valid
            obstacle_small = (dist_full > obstacle_thresh) & valid
            hole_small = (dist_full < -hole_thresh) & valid
            if max_above > 0.0:
                obstacle_small = obstacle_small & (dist_full <= max_above)

            h, w = frame.shape[:2]
            ground = cv2.resize(ground_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            obstacle = cv2.resize(obstacle_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            hole = cv2.resize(hole_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

            overlay = frame.copy()
            if controls["show_geom"]:
                overlay[ground == 1] = (0, 190, 0)
                overlay[obstacle == 1] = (0, 0, 255)
                overlay[hole == 1] = (255, 0, 0)
            vis = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)

            obstacle_mask = (obstacle * 255).astype(np.uint8)
            obstacle_mask = cv2.morphologyEx(
                obstacle_mask,
                cv2.MORPH_OPEN,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )
            obstacle_mask = cv2.morphologyEx(
                obstacle_mask,
                cv2.MORPH_CLOSE,
                np.ones((5, 5), dtype=np.uint8),
                iterations=1,
            )

            geom_boxes_for_ann: List[Tuple[int, int, int, int, str, str]] = []
            num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(obstacle_mask, 8)
            min_area = int(controls["min_box_area_px"])
            for idx in range(1, num_labels):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                x = int(stats[idx, cv2.CC_STAT_LEFT])
                y = int(stats[idx, cv2.CC_STAT_TOP])
                ww = int(stats[idx, cv2.CC_STAT_WIDTH])
                hh = int(stats[idx, cv2.CC_STAT_HEIGHT])
                x2 = x + ww
                y2 = y + hh
                geom_boxes_for_ann.append((x, y, x2, y2, "geom", "geom_obstacle"))
                if controls["show_boxes"]:
                    cv2.rectangle(vis, (x, y), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(
                        vis,
                        f"geom_obstacle area={area}",
                        (x, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            geom_count = len(geom_boxes_for_ann)

            ai_boxes_for_ann: List[Tuple[int, int, int, int, str, str]] = []
            ai_count = 0
            if controls["show_ai"] or annotation_mode:
                if yolo.available:
                    yolo_boxes = yolo.detect(
                        frame_bgr=frame,
                        conf_thresh=float(controls["ai_conf"]),
                        iou_thresh=float(controls["ai_iou"]),
                        frame_idx=frame_idx,
                    )
                    for x1, y1, x2, y2, conf, label in yolo_boxes:
                        ai_boxes_for_ann.append((x1, y1, x2, y2, "yolo", label))
                        if controls["show_ai"]:
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 255, 0), 2)
                            cv2.putText(
                                vis,
                                f"YOLO {label} {conf:.2f}",
                                (x1, max(16, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.52,
                                (180, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )
                if zed_detector.available:
                    zed_boxes = zed_detector.detect(
                        frame_idx=frame_idx,
                        conf_percent=int(max(1.0, min(99.0, controls["ai_conf"] * 100.0))),
                    )
                    for x1, y1, x2, y2, conf, label in zed_boxes:
                        ai_boxes_for_ann.append((x1, y1, x2, y2, "zed", label))
                        if controls["show_ai"]:
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 120), 2)
                            cv2.putText(
                                vis,
                                f"ZED {label} {conf:.2f}",
                                (x1, max(16, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.52,
                                (0, 255, 120),
                                1,
                                cv2.LINE_AA,
                            )
                ai_count = len(ai_boxes_for_ann)

            annotation_candidates = ai_boxes_for_ann + geom_boxes_for_ann

            if selected_class_idx >= 0:
                if selected_class_idx >= len(class_names):
                    print(f"No class configured for key {selected_class_idx + 1}")
                elif pending_box is None:
                    # No pending image box: numeric key still updates semantic brush class.
                    print(f"Brush class set to {class_names[selected_class_idx]} (no pending image box).")
                else:
                    saved_name = save_annotation(frame, pending_box, selected_class_idx, pending_source)
                    hit_count = project_box_to_semantic_map(
                        cloud_xyz=cloud,
                        box=pending_box,
                        class_id=selected_class_idx,
                        R_world_cam=R_world_cam,
                        t_world_cam=t_world_cam,
                    )
                    last_saved_msg = (
                        f"Saved {saved_name} class={class_names[selected_class_idx]} "
                        f"src={pending_source} map_hits={hit_count}"
                    )
                    print(last_saved_msg)
                    pending_box = None

            if pending_box is not None:
                x1, y1, x2, y2 = pending_box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(
                    vis,
                    f"pending [{pending_source}] press 1..{len(class_names)}",
                    (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            if annotation_mode and mouse_down and drag_start is not None and drag_curr is not None:
                sx, sy = drag_start
                ex, ey = drag_curr
                cv2.rectangle(vis, (sx, sy), (ex, ey), (255, 255, 255), 1)

            t_now = time.time()
            dt = max(1e-6, t_now - t_prev)
            t_prev = t_now
            fps = 1.0 / dt
            fps_smooth = fps if fps_smooth <= 0.0 else (0.9 * fps_smooth + 0.1 * fps)

            classes_hint = " ".join([f"{i + 1}:{name}" for i, name in enumerate(class_names)])
            status_lines = [
                f"FPS {fps_smooth:.1f} | stride {stride} | plane={'OK' if has_plane else 'WAIT'}",
                f"obs>{obstacle_thresh:.2f}m hole<{(-hole_thresh):.2f}m max_above={max_above:.2f}m max_fwd={controls['max_forward_m']:.1f}m",
                f"geom_boxes={geom_count} det_boxes={ai_count} det_mode={detector_mode} det={'ON' if controls['show_ai'] else 'OFF'} ann={'ON' if annotation_mode else 'OFF'}",
                "Keys: q=quit p=pause r=refloor s=snapshot l=annmode 0=clear",
                f"Label keys: {classes_hint}",
                f"Map brush: class={class_names[brush_class_idx]} | Semantic map: L-drag paint, R-click erase",
            ]
            if last_saved_msg:
                status_lines.append(last_saved_msg)
            ytxt = 24
            for line in status_lines:
                cv2.putText(vis, line, (12, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
                ytxt += 22

            cv2.imshow("Perception Lab", vis)
            cv2.imshow("Obstacle Mask", obstacle_mask)

            if args.semantic_map:
                cam_rc = None
                rr, cc, inb = _world_to_grid(
                    np.array([float(t_world_cam[0])]),
                    np.array([float(t_world_cam[2])]),
                    map_x_min,
                    map_z_min,
                    float(args.map_res_m),
                    map_h,
                    map_w,
                )
                if bool(inb[0]):
                    cam_rc = (int(rr[0]), int(cc[0]))
                sem_vis = _render_semantic_map(sem_counts, class_names, cam_rc)
                sem_show = cv2.resize(
                    sem_vis,
                    (sem_vis.shape[1] * map_display_scale, sem_vis.shape[0] * map_display_scale),
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.imshow("Semantic Map (XZ)", sem_show)

            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(SCRIPT_DIR, f"snapshot_{stamp}.png")
                cv2.imwrite(out_path, vis)
                print(f"Saved snapshot: {out_path}")

            frame_idx += 1
    finally:
        try:
            zed_detector.close()
        except Exception:
            pass
        try:
            zed.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
