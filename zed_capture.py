#!/usr/bin/env python3
import argparse
import os
import sys
import time
from datetime import datetime

import pyzed.sl as sl
import numpy as np
import cv2


def capture_one(out_path, width, height, warmup_frames, timeout_s):
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # good default for fast capture
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # faster since we're only taking a picture

    cam = sl.Camera()
    status = cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {repr(status)}")

    try:
        # Optional: set a custom output resolution (independent of camera mode)
        if width and height:
            cam.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)  # auto exposure on
            # We'll resize after capture for simplicity/reliability.

        runtime = sl.RuntimeParameters()
        image = sl.Mat()

        # Warm up a few frames so exposure/white balance settles
        for _ in range(max(0, warmup_frames)):
            cam.grab(runtime)

        start = time.time()
        while True:
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(image, sl.VIEW.LEFT)  # left RGB image

                # Convert to numpy (ZED gives BGRA)
                frame = image.get_data()
                if frame is None:
                    raise RuntimeError("Got empty frame data from ZED.")

                # BGRA -> BGR for OpenCV imwrite
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                if width and height:
                    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                ok = cv2.imwrite(out_path, bgr)
                if not ok:
                    raise RuntimeError(f"Failed to write image to: {out_path}")

                return out_path

            if (time.time() - start) > timeout_s:
                raise TimeoutError(f"Timed out waiting for a frame after {timeout_s} seconds: last error={repr(err)}")

    finally:
        cam.close()


def default_out_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"captures/zed_left_{ts}.png"


def main():
    p = argparse.ArgumentParser(description="Capture one image from a Stereolabs ZED camera.")
    p.add_argument("--out", default=None, help="Output image path (png/jpg). Default: captures/zed_left_<timestamp>.png")
    p.add_argument("--width", type=int, default=None, help="Optional output width (resize after capture).")
    p.add_argument("--height", type=int, default=None, help="Optional output height (resize after capture).")
    p.add_argument("--warmup", type=int, default=10, help="Warmup frames before saving (default: 10).")
    p.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for a frame (default: 5.0).")
    args = p.parse_args()

    out = args.out or default_out_path()

    try:
        saved = capture_one(out, args.width, args.height, args.warmup, args.timeout)
        print(saved)  # print path for easy scripting
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
