#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "ZEDAuto") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ZEDAuto"))


def _camera_source(value: str):
    stripped = str(value).strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def _open_capture(args: argparse.Namespace, cv2):
    source = _camera_source(args.device)
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    fourcc = args.fourcc.strip()
    if fourcc:
        if len(fourcc) != 4:
            raise ValueError("--fourcc must be exactly 4 characters, for example MJPG or YUYV")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open USB camera source {args.device!r}")

    return cap


def _draw_overlay(frame, args: argparse.Namespace, fps: float, cv2):
    if not args.overlay:
        return frame

    label = f"{args.source}  {args.device}  {fps:4.1f} FPS"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Display a USB camera on the Jetson/VNC screen.")
    parser.add_argument("--device", default=os.environ.get("USB_CAMERA_DEVICE", "0"))
    parser.add_argument("--width", type=int, default=int(os.environ.get("USB_CAMERA_WIDTH", "1280")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("USB_CAMERA_HEIGHT", "720")))
    parser.add_argument("--fps", type=int, default=int(os.environ.get("USB_CAMERA_FPS", "30")))
    parser.add_argument("--fourcc", default=os.environ.get("USB_CAMERA_FOURCC", "MJPG"))
    parser.add_argument("--window-name", default=os.environ.get("USB_CAMERA_WINDOW_NAME", "USB Camera"))
    parser.add_argument("--source", default=os.environ.get("USB_CAMERA_SOURCE", "usb_camera"))
    parser.add_argument("--flip", choices=("none", "horizontal", "vertical", "both"), default=os.environ.get("USB_CAMERA_FLIP", "none"))
    parser.add_argument("--fullscreen", action="store_true", default=os.environ.get("USB_CAMERA_FULLSCREEN", "0") == "1")
    parser.add_argument("--no-gui", action="store_true", default=os.environ.get("USB_CAMERA_NO_GUI", "0") == "1")
    parser.add_argument("--overlay", action="store_true", default=os.environ.get("USB_CAMERA_OVERLAY", "1") == "1")
    parser.add_argument("--publish-url", default=os.environ.get("USB_CAMERA_PUBLISH_URL", ""))
    parser.add_argument("--publish-interval-ms", type=int, default=int(os.environ.get("USB_CAMERA_PUBLISH_INTERVAL_MS", "120")))
    parser.add_argument("--publish-jpeg-quality", type=int, default=int(os.environ.get("USB_CAMERA_PUBLISH_JPEG_QUALITY", "75")))
    parser.add_argument("--heartbeat-url", default=os.environ.get("USB_CAMERA_HEARTBEAT_URL", ""))
    parser.add_argument("--heartbeat-interval-ms", type=int, default=int(os.environ.get("USB_CAMERA_HEARTBEAT_INTERVAL_MS", "1000")))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        import cv2
    except Exception as exc:
        print(f"OpenCV is required for USB camera viewing: {exc}", file=sys.stderr)
        return 2

    cap = _open_capture(args, cv2)

    publisher = None
    heartbeat = None
    if args.publish_url:
        from camera_publish_client import HttpCameraPublisher

        separator = "&" if "?" in args.publish_url else "?"
        publish_url = f"{args.publish_url}{separator}backend=opencv"
        publisher = HttpCameraPublisher(
            publish_url,
            interval_ms=args.publish_interval_ms,
            jpeg_quality=args.publish_jpeg_quality,
            source=args.source,
        )
    if args.heartbeat_url:
        from camera_status_client import CameraStatusHeartbeat

        heartbeat = CameraStatusHeartbeat(
            args.heartbeat_url,
            backend="opencv",
            source=args.source,
            interval_ms=args.heartbeat_interval_ms,
            streaming=True,
        )

    if not args.no_gui:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.setWindowProperty(args.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_t = time.monotonic()
    fps_ema = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"USB camera {args.device!r} returned no frame; retrying...")
                time.sleep(0.1)
                continue

            if args.flip == "horizontal":
                frame = cv2.flip(frame, 1)
            elif args.flip == "vertical":
                frame = cv2.flip(frame, 0)
            elif args.flip == "both":
                frame = cv2.flip(frame, -1)

            now = time.monotonic()
            instant_fps = 1.0 / max(now - last_t, 1e-6)
            fps_ema = instant_fps if fps_ema <= 0 else (0.9 * fps_ema + 0.1 * instant_fps)
            last_t = now

            vis = _draw_overlay(frame, args, fps_ema, cv2)

            if publisher is not None:
                publisher.push_frame(vis)

            if not args.no_gui:
                cv2.imshow(args.window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if heartbeat is not None:
            heartbeat.stop()
        if publisher is not None:
            publisher.stop()
        cap.release()
        if not args.no_gui:
            cv2.destroyWindow(args.window_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
