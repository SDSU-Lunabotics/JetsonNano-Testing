#!/usr/bin/env python3
"""
Unitree SDK -> JSONL bridge for lidar_ground_wall.py.

Outputs one JSON object per line to stdout:
  {"stamp": <unix_sec>, "frame_id": "unitree_l2", "points": [[x,y,z], ...]}

This bridge intentionally uses flexible reflection because Unitree Python API
varies across SDK wrappers/versions.
"""

import argparse
import importlib
import inspect
import json
import sys
import time
from typing import Any, Optional

import numpy as np


DEFAULT_MODULE_CANDIDATES = [
    "unitree_lidar_sdk2",
    "unitree_lidar_sdk",
    "unitree_lidar",
    "unilidar_sdk",
]

DEFAULT_CLASS_CANDIDATES = [
    "UnitreeLidar", "Lidar", "LidarSDK", "L2", "L2Lidar", "Driver", "LidarDriver"
]

DEFAULT_READ_METHODS = [
    "read_points",
    "get_points",
    "get_point_cloud",
    "get_pointcloud",
    "poll",
    "read",
    "next_frame",
    "grab",
]

DEFAULT_START_METHODS = ["start", "connect", "initialize", "init", "open", "run"]
DEFAULT_STOP_METHODS = ["stop", "close", "shutdown", "disconnect"]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _pick_module(module_name: str):
    if module_name:
        return importlib.import_module(module_name)

    last_exc = None
    for name in DEFAULT_MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        "Could not import Unitree Python SDK module. "
        f"Tried: {DEFAULT_MODULE_CANDIDATES}. Last error: {last_exc}"
    )


def _call_with_supported_kwargs(fn, kwargs: dict):
    try:
        sig = inspect.signature(fn)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
        return fn(**allowed)
    except Exception:
        # Fallback: try no kwargs.
        return fn()


def _coerce_xyz(points: Any) -> np.ndarray:
    if points is None:
        return np.empty((0, 3), dtype=np.float32)

    # dict formats
    if isinstance(points, dict):
        if "points" in points:
            return _coerce_xyz(points["points"])
        if "xyz" in points:
            return _coerce_xyz(points["xyz"])
        if "x" in points and "y" in points and "z" in points:
            x = np.asarray(points["x"], dtype=np.float32).reshape(-1)
            y = np.asarray(points["y"], dtype=np.float32).reshape(-1)
            z = np.asarray(points["z"], dtype=np.float32).reshape(-1)
            n = min(x.size, y.size, z.size)
            if n <= 0:
                return np.empty((0, 3), dtype=np.float32)
            arr = np.stack((x[:n], y[:n], z[:n]), axis=1)
            finite = np.isfinite(arr).all(axis=1)
            return arr[finite]

    # numpy-like
    if hasattr(points, "shape"):
        arr = np.asarray(points, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                raise ValueError("flat points array length must be divisible by 3")
            arr = arr.reshape(-1, 3)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("points must be Nx3")
        arr = arr[:, :3]
        finite = np.isfinite(arr).all(axis=1)
        return arr[finite]

    # iterable of tuples/lists/objects
    if isinstance(points, (list, tuple)):
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        first = points[0]
        if hasattr(first, "x") and hasattr(first, "y") and hasattr(first, "z"):
            arr = np.array([[float(p.x), float(p.y), float(p.z)] for p in points], dtype=np.float32)
            finite = np.isfinite(arr).all(axis=1)
            return arr[finite]

        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                raise ValueError("flat points list length must be divisible by 3")
            arr = arr.reshape(-1, 3)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("points list must be Nx3")
        arr = arr[:, :3]
        finite = np.isfinite(arr).all(axis=1)
        return arr[finite]

    raise ValueError(f"Unsupported point payload type: {type(points)}")


class GenericUnitreeAdapter:
    def __init__(self, module, class_name: str, kwargs: dict):
        self.module = module
        self.driver = self._build_driver(class_name, kwargs)
        self.read_method = self._find_read_method()
        self.stop_method = self._find_stop_method()

    def _build_driver(self, class_name: str, kwargs: dict):
        # 1) factory function patterns
        for factory_name in ("create_driver", "create_lidar", "make_driver"):
            if hasattr(self.module, factory_name):
                fn = getattr(self.module, factory_name)
                drv = _call_with_supported_kwargs(fn, kwargs)
                self._start_driver(drv, kwargs)
                return drv

        # 2) class patterns
        class_candidates = [class_name] if class_name else []
        class_candidates.extend(DEFAULT_CLASS_CANDIDATES)

        last_exc = None
        for name in class_candidates:
            if not name:
                continue
            if not hasattr(self.module, name):
                continue
            cls = getattr(self.module, name)
            try:
                drv = _call_with_supported_kwargs(cls, kwargs)
                self._start_driver(drv, kwargs)
                return drv
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(
            f"Could not construct Unitree driver from module '{self.module.__name__}'. "
            f"Class candidates: {class_candidates}. Last error: {last_exc}"
        )

    def _start_driver(self, drv, kwargs: dict):
        for name in DEFAULT_START_METHODS:
            if hasattr(drv, name):
                fn = getattr(drv, name)
                try:
                    _call_with_supported_kwargs(fn, kwargs)
                    return
                except Exception:
                    # Try no args directly once.
                    try:
                        fn()
                        return
                    except Exception:
                        continue

    def _find_read_method(self):
        for name in DEFAULT_READ_METHODS:
            if hasattr(self.driver, name):
                return getattr(self.driver, name)
        raise RuntimeError(
            f"No supported read method found. Tried: {DEFAULT_READ_METHODS}. "
            "Edit unitree_sdk_bridge.py for your SDK API."
        )

    def _find_stop_method(self):
        for name in DEFAULT_STOP_METHODS:
            if hasattr(self.driver, name):
                return getattr(self.driver, name)
        return None

    def read_xyz(self) -> np.ndarray:
        payload = self.read_method()

        # Some APIs return (ok, data)
        if isinstance(payload, tuple) and len(payload) == 2:
            ok, data = payload
            if isinstance(ok, bool) and not ok:
                return np.empty((0, 3), dtype=np.float32)
            payload = data

        # Some APIs return object with common fields
        if hasattr(payload, "points"):
            payload = payload.points
        elif hasattr(payload, "xyz"):
            payload = payload.xyz

        return _coerce_xyz(payload)

    def close(self):
        if self.stop_method is None:
            return
        try:
            self.stop_method()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser(description="Unitree SDK JSONL bridge")
    p.add_argument("--module", default="", help="Python module name for Unitree SDK wrapper")
    p.add_argument("--class-name", default="", help="Driver class name in module")
    p.add_argument("--frame-id", default="unitree_l2", help="frame_id written to JSON output")
    p.add_argument("--hz", type=float, default=20.0, help="Read/publish loop rate")
    p.add_argument("--log-every-sec", type=float, default=2.0, help="Status print period")

    # Optional constructor/start kwargs (passed when supported by SDK API)
    p.add_argument("--device", default="", help="Device path, if SDK expects one")
    p.add_argument("--ip", default="", help="LiDAR IP, if SDK expects one")
    p.add_argument("--port", type=int, default=0, help="LiDAR port, if SDK expects one")
    p.add_argument("--scan-frequency", type=float, default=0.0, help="Scan frequency hint")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    module = _pick_module(args.module)
    _log(f"Using SDK module: {module.__name__}")

    kwargs = {
        "device": args.device or None,
        "ip": args.ip or None,
        "port": (int(args.port) if int(args.port) > 0 else None),
        "scan_frequency": (float(args.scan_frequency) if float(args.scan_frequency) > 0.0 else None),
    }

    adapter = GenericUnitreeAdapter(module, args.class_name, kwargs)
    _log("Unitree adapter initialized. Streaming JSON frames...")

    period = 1.0 / max(1.0, float(args.hz))
    next_tick = time.monotonic()
    last_log = 0.0
    frame_count = 0

    try:
        while True:
            xyz = adapter.read_xyz()
            stamp = time.time()

            payload = {
                "stamp": stamp,
                "frame_id": args.frame_id,
                "points": xyz.tolist(),
            }
            sys.stdout.write(json.dumps(payload) + "\n")
            sys.stdout.flush()

            frame_count += 1
            now = time.monotonic()
            if (now - last_log) >= max(0.1, float(args.log_every_sec)):
                _log(f"bridge frames={frame_count} points={xyz.shape[0]}")
                last_log = now

            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        pass
    finally:
        adapter.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
