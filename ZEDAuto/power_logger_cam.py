#!/usr/bin/env python3
import argparse
import os
import time
from datetime import datetime

import cv2


WINDOW_NAME = "Power Logger Camera"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Preview and optionally record a USB camera for power logger monitoring."
    )
    parser.add_argument("--device", default="/dev/video0", help="Video device path.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested capture FPS.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "recordings", "power_logger"),
        help="Directory where recordings are saved.",
    )
    parser.add_argument(
        "--auto-record",
        action="store_true",
        help="Start recording immediately.",
    )
    return parser


def open_camera(device, width, height, fps):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def make_writer(output_dir, frame_width, frame_height, fps):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"power_logger_{timestamp}.avi")
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output file {path}")
    return writer, path


def draw_overlay(frame, recording, output_path, frame_width, frame_height, fps):
    status_color = (0, 220, 0) if recording else (0, 180, 255)
    status_text = "REC" if recording else "PREVIEW"
    lines = [
        f"{status_text}  {frame_width}x{frame_height} @ {fps:.1f} fps",
        "Keys: r = start/stop recording, q or ESC = quit",
    ]
    if output_path:
        lines.append(f"File: {os.path.basename(output_path)}")

    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color if line == lines[0] else (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        y += 32


def main():
    args = build_arg_parser().parse_args()

    cap = open_camera(args.device, args.width, args.height, args.fps)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps

    writer = None
    output_path = None
    recording = False

    if args.auto_record:
        writer, output_path = make_writer(args.output_dir, actual_width, actual_height, actual_fps)
        recording = True
        print(f"Recording started: {output_path}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(actual_width, 1280), min(actual_height, 720))

    last_read_failure = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                now = time.time()
                if now - last_read_failure > 1.0:
                    print("Warning: camera frame read failed")
                    last_read_failure = now
                time.sleep(0.02)
                continue

            if recording and writer is not None:
                writer.write(frame)

            preview = frame.copy()
            draw_overlay(preview, recording, output_path, actual_width, actual_height, actual_fps)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                if recording:
                    recording = False
                    assert writer is not None
                    writer.release()
                    writer = None
                    print(f"Recording stopped: {output_path}")
                    output_path = None
                else:
                    writer, output_path = make_writer(
                        args.output_dir, actual_width, actual_height, actual_fps
                    )
                    recording = True
                    print(f"Recording started: {output_path}")
    finally:
        if writer is not None:
            writer.release()
            print(f"Recording saved: {output_path}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
