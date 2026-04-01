#!/usr/bin/env python3
"""
Unitree L2 LiDAR ground/wall/hole segmentation + 2D occupancy map.

Supports two input modes:
- sdk: direct Unitree SDK bridge process (no ROS2 required)
- ros2: ROS2 PointCloud2 topic
"""

import argparse
import json
import shlex
import select
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from lidar_map_utils import OccupancyMap

try:
    import rclpy
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import PointCloud2
except Exception:
    rclpy = None
    QoSProfile = None
    PointCloud2 = None

try:
    from sensor_msgs_py import point_cloud2 as ros_pc2
except Exception:
    ros_pc2 = None


AXIS_TO_IDX = {"x": 0, "y": 1, "z": 2}


@dataclass
class PlaneState:
    normal: np.ndarray
    d: float
    inlier_ratio: float
    valid: bool


class Ros2PointCloudSource:
    source_name = "ros2"

    def __init__(self, topic: str, queue_size: int = 5, frame_stride: int = 1):
        if rclpy is None or PointCloud2 is None:
            raise RuntimeError(
                "ROS2 Python modules are missing. Install ROS2 Python + sensor_msgs on this machine."
            )

        if not rclpy.ok():
            rclpy.init(args=None)

        self.topic = topic
        self.frame_stride = max(1, int(frame_stride))
        self.frames_received = 0
        self.last_error = ""
        self._latest = None

        self.node = rclpy.create_node("unitree_l2_ground_wall_map")
        qos = QoSProfile(depth=max(1, int(queue_size)))
        self.sub = self.node.create_subscription(PointCloud2, self.topic, self._on_msg, qos)

    def _on_msg(self, msg: PointCloud2) -> None:
        self.frames_received += 1
        if (self.frames_received - 1) % self.frame_stride != 0:
            return

        try:
            xyz = pointcloud2_to_xyz(msg)
        except Exception as exc:
            self.last_error = f"point cloud decode failed: {exc}"
            return

        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if stamp <= 0.0:
            stamp = time.time()

        self._latest = (stamp, xyz, msg.header.frame_id)

    def poll(self, timeout_sec: float = 0.05):
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while True:
            rclpy.spin_once(self.node, timeout_sec=0.01)
            if self._latest is not None:
                out = self._latest
                self._latest = None
                return out
            if time.monotonic() >= deadline:
                return None

    def close(self) -> None:
        try:
            if self.node is not None:
                self.node.destroy_node()
        finally:
            if rclpy is not None and rclpy.ok():
                rclpy.shutdown()


class SdkJsonlCommandSource:
    """Reads newline-delimited JSON frames from a subprocess.

    Expected JSON line format (one frame per line):
      {"stamp": 1710000000.123, "frame_id": "unitree_l2", "points": [[x,y,z], ...]}

    Also accepts:
      {"x": [...], "y": [...], "z": [...]}  # equal-length arrays
      {"xyz": [[x,y,z], ...]}
    """

    source_name = "sdk"

    def __init__(self, cmd: str, frame_stride: int = 1, startup_timeout_sec: float = 5.0):
        if not cmd or not str(cmd).strip():
            raise RuntimeError("SDK mode requires --sdk-cmd")

        self.cmd = str(cmd)
        self.frame_stride = max(1, int(frame_stride))
        self.frames_received = 0
        self.last_error = ""
        self._stderr_buf = ""

        argv = shlex.split(self.cmd)
        if not argv:
            raise RuntimeError("Invalid --sdk-cmd (empty after parsing)")

        self.proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Give the bridge a short startup window to fail fast with useful logs.
        deadline = time.monotonic() + max(0.1, float(startup_timeout_sec))
        while time.monotonic() < deadline:
            self._drain_stderr_nonblocking()
            if self.proc.poll() is not None:
                err = self._stderr_buf.strip()
                raise RuntimeError(
                    "SDK bridge exited early (%s)%s"
                    % (
                        str(self.proc.returncode),
                        (": " + err[-400:]) if err else "",
                    )
                )
            # If bridge prints nothing but stays alive, that is fine.
            time.sleep(0.02)

    def _drain_stderr_nonblocking(self) -> None:
        if self.proc.stderr is None:
            return
        try:
            while True:
                ready, _, _ = select.select([self.proc.stderr], [], [], 0.0)
                if not ready:
                    break
                line = self.proc.stderr.readline()
                if not line:
                    break
                self._stderr_buf += line
                self.last_error = line.strip()
        except Exception:
            pass

    @staticmethod
    def _coerce_xyz(value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                raise ValueError("flat point array length must be divisible by 3")
            arr = arr.reshape(-1, 3)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("point array must be Nx3")
        xyz = arr[:, :3]
        finite = np.isfinite(xyz).all(axis=1)
        return xyz[finite]

    def _parse_line(self, line: str):
        line = line.strip()
        if not line:
            return None

        payload = json.loads(line)

        stamp = time.time()
        frame_id = "unitree_l2"

        if isinstance(payload, dict):
            if "stamp" in payload:
                try:
                    stamp = float(payload["stamp"])
                except Exception:
                    pass
            if "frame_id" in payload and payload["frame_id"] is not None:
                frame_id = str(payload["frame_id"])

            if "points" in payload:
                xyz = self._coerce_xyz(payload["points"])
            elif "xyz" in payload:
                xyz = self._coerce_xyz(payload["xyz"])
            elif "x" in payload and "y" in payload and "z" in payload:
                x = np.asarray(payload["x"], dtype=np.float32).reshape(-1)
                y = np.asarray(payload["y"], dtype=np.float32).reshape(-1)
                z = np.asarray(payload["z"], dtype=np.float32).reshape(-1)
                n = min(x.size, y.size, z.size)
                xyz = np.stack((x[:n], y[:n], z[:n]), axis=1) if n > 0 else np.empty((0, 3), dtype=np.float32)
                finite = np.isfinite(xyz).all(axis=1)
                xyz = xyz[finite]
            else:
                raise ValueError("JSON frame missing points/xyz or x,y,z")

            return stamp, xyz, frame_id

        if isinstance(payload, list):
            xyz = self._coerce_xyz(payload)
            return stamp, xyz, frame_id

        raise ValueError("Unsupported JSON payload type")

    def poll(self, timeout_sec: float = 0.05):
        if self.proc.poll() is not None:
            self._drain_stderr_nonblocking()
            err = self._stderr_buf.strip()
            self.last_error = (
                "SDK bridge exited (%s)%s"
                % (
                    str(self.proc.returncode),
                    (": " + err[-400:]) if err else "",
                )
            )
            return None

        if self.proc.stdout is None:
            self.last_error = "SDK bridge missing stdout"
            return None

        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while True:
            self._drain_stderr_nonblocking()
            remain = deadline - time.monotonic()
            if remain <= 0.0:
                return None

            ready, _, _ = select.select([self.proc.stdout], [], [], remain)
            if not ready:
                return None

            line = self.proc.stdout.readline()
            if not line:
                # EOF or no data.
                if self.proc.poll() is not None:
                    self._drain_stderr_nonblocking()
                    err = self._stderr_buf.strip()
                    self.last_error = (
                        "SDK bridge ended (%s)%s"
                        % (
                            str(self.proc.returncode),
                            (": " + err[-400:]) if err else "",
                        )
                    )
                return None

            try:
                parsed = self._parse_line(line)
            except Exception as exc:
                self.last_error = f"SDK JSON parse failed: {exc}"
                continue

            if parsed is None:
                continue

            self.frames_received += 1
            if (self.frames_received - 1) % self.frame_stride != 0:
                continue
            return parsed

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except Exception:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=0.5)
                except Exception:
                    pass
        self._drain_stderr_nonblocking()


def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    if ros_pc2 is not None:
        pts = np.asarray(
            list(ros_pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32,
        )
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[1] < 3:
            raise RuntimeError("PointCloud2 is missing x,y,z fields")
        return pts[:, :3]

    # Fallback when sensor_msgs_py is unavailable.
    return pointcloud2_to_xyz_fallback(msg)


def pointcloud2_to_xyz_fallback(msg: PointCloud2) -> np.ndarray:
    field_offsets = {}
    field_types = {}
    for field in msg.fields:
        field_offsets[field.name] = int(field.offset)
        field_types[field.name] = int(field.datatype)

    for name in ("x", "y", "z"):
        if name not in field_offsets:
            raise RuntimeError(f"PointCloud2 missing '{name}' field")
        # datatype 7 is FLOAT32 in ROS PointField.
        if field_types[name] != 7:
            raise RuntimeError("Fallback decoder only supports FLOAT32 x/y/z fields")

    step = int(msg.point_step)
    total_points = int(msg.width) * int(msg.height)
    if step <= 0 or total_points <= 0 or len(msg.data) == 0:
        return np.empty((0, 3), dtype=np.float32)

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    total_points = min(total_points, raw.size // step)
    if total_points <= 0:
        return np.empty((0, 3), dtype=np.float32)

    raw = raw[: total_points * step].reshape(total_points, step)

    def extract_float(offset: int) -> np.ndarray:
        chunk = raw[:, offset : offset + 4]
        return chunk.view(np.float32).reshape(-1)

    x = extract_float(field_offsets["x"])
    y = extract_float(field_offsets["y"])
    z = extract_float(field_offsets["z"])

    xyz = np.stack((x, y, z), axis=1).astype(np.float32, copy=False)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite]


def parse_args():
    p = argparse.ArgumentParser(description="Unitree L2 LiDAR ground/wall map")

    # Input source
    p.add_argument(
        "--input-mode",
        choices=["sdk", "ros2", "auto"],
        default="sdk",
        help="Point source: direct SDK bridge, ROS2 topic, or auto-try sdk->ros2",
    )
    p.add_argument(
        "--sdk-cmd",
        default="python3 ./lidar-testing/unitree_sdk_bridge.py",
        help="Command that outputs newline JSON frames for SDK mode",
    )
    p.add_argument(
        "--sdk-startup-timeout-sec",
        type=float,
        default=5.0,
        help="How long to wait for SDK bridge startup before failing",
    )

    # ROS2 mode
    p.add_argument("--topic", default="/utlidar/cloud", help="ROS2 PointCloud2 topic")
    p.add_argument("--queue-size", type=int, default=5, help="ROS2 subscription queue depth")

    # Common input options
    p.add_argument("--frame-stride", type=int, default=1, help="Only process every Nth cloud frame")
    p.add_argument("--stride", type=int, default=2, help="Point stride inside each cloud (performance)")
    p.add_argument("--up-axis", choices=["x", "y", "z"], default="z", help="Axis that points upward")
    p.add_argument(
        "--forward-axis",
        choices=["x", "y", "z"],
        default="x",
        help="Axis that points forward for map display",
    )
    p.add_argument(
        "--lateral-axis",
        choices=["x", "y", "z"],
        default="y",
        help="Axis that points left/right for map display",
    )

    # Point filtering
    p.add_argument("--min-range-m", type=float, default=0.20, help="Ignore points closer than this")
    p.add_argument("--max-range-m", type=float, default=12.0, help="Ignore points farther than this")
    p.add_argument("--min-forward-m", type=float, default=-1.0, help="Min forward distance to keep")
    p.add_argument("--max-forward-m", type=float, default=12.0, help="Max forward distance to keep")
    p.add_argument("--max-abs-lateral-m", type=float, default=10.0, help="Max abs lateral distance to keep")

    # Ground / wall / hole segmentation
    p.add_argument("--ground-thresh-m", type=float, default=0.08, help="Distance from plane treated as ground")
    p.add_argument("--obstacle-thresh-m", type=float, default=0.08, help="Above-plane threshold for wall/obstacle")
    p.add_argument("--hole-thresh-m", type=float, default=0.10, help="Below-plane threshold for holes")
    p.add_argument(
        "--max-above-ground-m",
        type=float,
        default=1.22,
        help="Ignore points above this plane-relative height (0 to disable)",
    )
    p.add_argument("--disable-holes", action="store_true", help="Disable hole classification")

    # Plane fitting
    p.add_argument("--plane-update-sec", type=float, default=0.50, help="Seconds between plane re-fit")
    p.add_argument("--plane-ransac-iters", type=int, default=120, help="RANSAC iterations for floor plane")
    p.add_argument(
        "--plane-min-normal-up",
        type=float,
        default=0.70,
        help="Min up-axis component for a valid floor normal",
    )
    p.add_argument(
        "--plane-fit-min-range-m",
        type=float,
        default=0.25,
        help="Min range for points used in plane fit",
    )
    p.add_argument(
        "--plane-fit-max-range-m",
        type=float,
        default=5.0,
        help="Max range for points used in plane fit",
    )
    p.add_argument(
        "--plane-fit-max-abs-up-m",
        type=float,
        default=0.60,
        help="Use only points with |up| below this for plane fit",
    )

    # Map
    p.add_argument("--map-width-m", type=float, default=20.0, help="Map width in meters (lateral axis)")
    p.add_argument("--map-height-m", type=float, default=20.0, help="Map height in meters (forward axis)")
    p.add_argument("--map-res-m", type=float, default=0.05, help="Map resolution in meters per cell")
    p.add_argument(
        "--map-forward-min",
        type=float,
        default=0.0,
        help="Forward min bound when not centering map",
    )
    p.add_argument("--map-center", action="store_true", help="Center map around rover forward=0")
    p.add_argument("--map-scale", type=int, default=2, help="Display scale for map window")

    p.add_argument("--map-save-path", default="lidar_map.npz", help="Path for persistent map file")
    p.add_argument("--map-save-every", type=float, default=5.0, help="Autosave period in seconds (0 disables)")
    p.add_argument("--map-load", action="store_true", help="Load map file on startup if present")

    p.add_argument("--free-decay", type=float, default=0.995, help="Free-space decay")
    p.add_argument("--occ-decay", type=float, default=0.98, help="Obstacle decay")
    p.add_argument("--hole-decay", type=float, default=0.98, help="Hole decay")

    p.add_argument("--print-every-sec", type=float, default=1.0, help="Console status print period")
    p.add_argument("--no-gui", action="store_true", help="Disable OpenCV windows")

    return p.parse_args()


def ensure_axis_triplet(up_axis: str, forward_axis: str, lateral_axis: str) -> None:
    axes = [up_axis, forward_axis, lateral_axis]
    if len(set(axes)) != 3:
        raise ValueError("up-axis, forward-axis, and lateral-axis must all be different")


def fit_plane_from_inliers(points_xyz: np.ndarray):
    centroid = points_xyz.mean(axis=0)
    centered = points_xyz - centroid
    cov = centered.T @ centered / max(1, (points_xyz.shape[0] - 1))
    eigvals, eigvecs = np.linalg.eigh(cov)
    _ = eigvals
    normal = eigvecs[:, 0]
    d = -float(np.dot(normal, centroid))
    return normal.astype(np.float32), d


def fit_ground_plane_ransac(
    points_xyz: np.ndarray,
    up_axis_idx: int,
    distance_thresh: float,
    min_normal_up: float,
    iterations: int,
) -> PlaneState:
    n = points_xyz.shape[0]
    if n < 50:
        return PlaneState(np.zeros(3, dtype=np.float32), 0.0, 0.0, False)

    rng = np.random.default_rng()
    best_count = 0
    best_inliers = None

    for _ in range(max(20, int(iterations))):
        sample_idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points_xyz[sample_idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-7:
            continue

        normal = normal / norm
        if normal[up_axis_idx] < 0.0:
            normal = -normal
        if normal[up_axis_idx] < min_normal_up:
            continue

        d = -float(np.dot(normal, p0))
        dist = points_xyz @ normal + d
        inliers = np.abs(dist) <= distance_thresh
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 30:
        return PlaneState(np.zeros(3, dtype=np.float32), 0.0, 0.0, False)

    inlier_points = points_xyz[best_inliers]
    normal, d = fit_plane_from_inliers(inlier_points)
    if normal[up_axis_idx] < 0.0:
        normal = -normal
        d = -d

    if normal[up_axis_idx] < min_normal_up:
        return PlaneState(np.zeros(3, dtype=np.float32), 0.0, 0.0, False)

    dist_all = points_xyz @ normal + d
    inlier_ratio = float(np.count_nonzero(np.abs(dist_all) <= distance_thresh)) / float(n)
    return PlaneState(normal.astype(np.float32), float(d), inlier_ratio, True)


def draw_robot_marker(map_vis: np.ndarray, robot_rc: Optional[tuple], size_px: int = 5) -> None:
    if robot_rc is None:
        return
    rr, cc = robot_rc
    half = max(1, int(size_px) // 2)
    h, w = map_vis.shape[:2]
    r1 = max(0, rr - half)
    r2 = min(h, rr + half + 1)
    c1 = max(0, cc - half)
    c2 = min(w, cc + half + 1)
    map_vis[r1:r2, c1:c2] = (255, 0, 0)

    tip_r = max(0, rr - max(10, size_px * 2))
    cv2.arrowedLine(map_vis, (cc, rr), (cc, tip_r), (255, 255, 0), 1, cv2.LINE_AA, tipLength=0.35)


def overlay_status(
    map_vis: np.ndarray,
    source_name: str,
    source_target: str,
    frame_id: str,
    points_count: int,
    ground_pct: float,
    obstacle_pct: float,
    hole_pct: float,
    plane: PlaneState,
    last_plane_age: float,
) -> np.ndarray:
    panel_h = 138
    _, w = map_vis.shape[:2]
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)

    def put(txt: str, y: int, color=(230, 230, 230)):
        cv2.putText(panel, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    put("UNITREE L2 GROUND/WALL MAP", 20, (200, 255, 255))
    put(f"Source: {source_name} | {source_target}", 42, (170, 240, 255))
    put(f"Frame: {frame_id or 'unknown'} | Points: {points_count}", 64)
    put(f"Ground {ground_pct:5.1f}% | Obstacles {obstacle_pct:5.1f}% | Holes {hole_pct:5.1f}%", 86)

    if plane.valid:
        n = plane.normal
        put(
            "Plane n=(%+.2f,%+.2f,%+.2f) d=%+.2f inliers=%4.1f%% age=%4.1fs"
            % (n[0], n[1], n[2], plane.d, plane.inlier_ratio * 100.0, last_plane_age),
            108,
            (120, 220, 255),
        )
    else:
        put("Plane: unavailable (waiting for enough points)", 108, (80, 180, 255))

    put("Keys: q/esc quit, s save map", 130, (180, 255, 180))
    return np.vstack((panel, map_vis))


def make_source(args):
    if args.input_mode == "ros2":
        source = Ros2PointCloudSource(args.topic, queue_size=args.queue_size, frame_stride=args.frame_stride)
        return source, "ros2", args.topic

    if args.input_mode == "sdk":
        source = SdkJsonlCommandSource(
            args.sdk_cmd,
            frame_stride=args.frame_stride,
            startup_timeout_sec=args.sdk_startup_timeout_sec,
        )
        return source, "sdk", args.sdk_cmd

    # auto: prefer sdk first, then ros2.
    sdk_err = None
    try:
        source = SdkJsonlCommandSource(
            args.sdk_cmd,
            frame_stride=args.frame_stride,
            startup_timeout_sec=args.sdk_startup_timeout_sec,
        )
        return source, "sdk", args.sdk_cmd
    except Exception as exc:
        sdk_err = exc

    try:
        source = Ros2PointCloudSource(args.topic, queue_size=args.queue_size, frame_stride=args.frame_stride)
        print(f"SDK source unavailable, falling back to ROS2. SDK error: {sdk_err}")
        return source, "ros2", args.topic
    except Exception as ros_exc:
        raise RuntimeError(f"SDK failed ({sdk_err}) and ROS2 failed ({ros_exc})")


def main() -> int:
    args = parse_args()
    ensure_axis_triplet(args.up_axis, args.forward_axis, args.lateral_axis)

    up_idx = AXIS_TO_IDX[args.up_axis]
    fwd_idx = AXIS_TO_IDX[args.forward_axis]
    lat_idx = AXIS_TO_IDX[args.lateral_axis]

    map_forward_min = -args.map_height_m / 2.0 if args.map_center else args.map_forward_min
    occ_map = OccupancyMap(
        map_res_m=args.map_res_m,
        map_width_m=args.map_width_m,
        map_height_m=args.map_height_m,
        map_forward_min=map_forward_min,
        free_decay=args.free_decay,
        occ_decay=args.occ_decay,
        hole_decay=args.hole_decay,
    )

    if args.map_load:
        try:
            ok, msg = occ_map.load(args.map_save_path)
            print(f"{msg}: {args.map_save_path}" if ok else msg)
        except Exception as exc:
            print(f"Map load failed ({args.map_save_path}): {exc}")

    source, source_name, source_target = make_source(args)

    plane = PlaneState(normal=np.zeros(3, dtype=np.float32), d=0.0, inlier_ratio=0.0, valid=False)
    plane.normal[up_idx] = 1.0

    last_plane_fit_t = 0.0
    last_print_t = 0.0
    last_save_t = time.time()
    last_frame_id = ""
    last_points_count = 0
    last_ground_pct = 0.0
    last_obstacle_pct = 0.0
    last_hole_pct = 0.0

    print(f"Input source: {source_name} | target: {source_target}")
    print(
        f"Axis mapping -> up:{args.up_axis} forward:{args.forward_axis} lateral:{args.lateral_axis} | "
        f"map center={'ON' if args.map_center else 'OFF'}"
    )

    try:
        while True:
            polled = source.poll(timeout_sec=0.05)
            if polled is not None:
                _stamp, xyz, frame_id = polled
                last_frame_id = frame_id

                if xyz.size > 0 and args.stride > 1:
                    xyz = xyz[:: int(args.stride)]
                if xyz.size > 0:
                    finite = np.isfinite(xyz).all(axis=1)
                    xyz = xyz[finite]

                if xyz.shape[0] > 0:
                    lat = xyz[:, lat_idx]
                    fwd = xyz[:, fwd_idx]

                    planar_range = np.hypot(lat, fwd)
                    keep = (
                        (planar_range >= args.min_range_m)
                        & (planar_range <= args.max_range_m)
                        & (fwd >= args.min_forward_m)
                        & (fwd <= args.max_forward_m)
                        & (np.abs(lat) <= args.max_abs_lateral_m)
                    )

                    xyz = xyz[keep]
                    if xyz.shape[0] > 0:
                        lat = xyz[:, lat_idx]
                        fwd = xyz[:, fwd_idx]
                        up = xyz[:, up_idx]

                        now = time.time()
                        if (not plane.valid) or ((now - last_plane_fit_t) >= args.plane_update_sec):
                            fit_range = np.hypot(lat, fwd)
                            fit_mask = (
                                (fit_range >= args.plane_fit_min_range_m)
                                & (fit_range <= args.plane_fit_max_range_m)
                                & (np.abs(up) <= args.plane_fit_max_abs_up_m)
                            )
                            fit_xyz = xyz[fit_mask]
                            if fit_xyz.shape[0] >= 80:
                                candidate = fit_ground_plane_ransac(
                                    fit_xyz,
                                    up_axis_idx=up_idx,
                                    distance_thresh=args.ground_thresh_m,
                                    min_normal_up=args.plane_min_normal_up,
                                    iterations=args.plane_ransac_iters,
                                )
                                if candidate.valid:
                                    plane = candidate
                                    last_plane_fit_t = now

                        if plane.valid:
                            dist = xyz @ plane.normal + plane.d
                        else:
                            # Fallback until we have a stable plane.
                            dist = xyz[:, up_idx]

                        if args.max_above_ground_m > 0.0:
                            keep_height = dist <= float(args.max_above_ground_m)
                            xyz = xyz[keep_height]
                            dist = dist[keep_height]
                            if xyz.shape[0] == 0:
                                continue
                            lat = xyz[:, lat_idx]
                            fwd = xyz[:, fwd_idx]

                        ground_mask = np.abs(dist) <= args.ground_thresh_m
                        obstacle_mask = dist > args.obstacle_thresh_m
                        if args.disable_holes:
                            hole_mask = np.zeros_like(ground_mask, dtype=bool)
                        else:
                            hole_mask = dist < -args.hole_thresh_m

                        occ_map.update(lat, fwd, ground_mask, obstacle_mask, hole_mask)

                        denom = float(max(1, xyz.shape[0]))
                        last_points_count = int(xyz.shape[0])
                        last_ground_pct = 100.0 * float(np.count_nonzero(ground_mask)) / denom
                        last_obstacle_pct = 100.0 * float(np.count_nonzero(obstacle_mask)) / denom
                        last_hole_pct = 100.0 * float(np.count_nonzero(hole_mask)) / denom

            now = time.time()

            if now - last_print_t >= max(0.1, args.print_every_sec):
                last_print_t = now
                plane_age = now - last_plane_fit_t if last_plane_fit_t > 0.0 else -1.0
                print(
                    "LiDAR src=%s frames=%d points=%d ground=%5.1f%% obstacles=%5.1f%% holes=%5.1f%% plane_ok=%s age=%.2fs"
                    % (
                        source_name,
                        source.frames_received,
                        last_points_count,
                        last_ground_pct,
                        last_obstacle_pct,
                        last_hole_pct,
                        str(plane.valid),
                        plane_age,
                    )
                )
                if source.last_error:
                    print(f"Source warning: {source.last_error}")
                    source.last_error = ""

            if args.map_save_every > 0.0 and (now - last_save_t) >= args.map_save_every:
                try:
                    occ_map.save(args.map_save_path)
                    last_save_t = now
                except Exception as exc:
                    print(f"Map save failed ({args.map_save_path}): {exc}")

            if not args.no_gui:
                map_vis = occ_map.render()
                robot_rc = occ_map.world_to_grid(0.0, 0.0)
                draw_robot_marker(map_vis, robot_rc, size_px=5)
                plane_age = now - last_plane_fit_t if last_plane_fit_t > 0.0 else -1.0
                disp = overlay_status(
                    map_vis,
                    source_name=source_name,
                    source_target=source_target,
                    frame_id=last_frame_id,
                    points_count=last_points_count,
                    ground_pct=last_ground_pct,
                    obstacle_pct=last_obstacle_pct,
                    hole_pct=last_hole_pct,
                    plane=plane,
                    last_plane_age=plane_age,
                )

                if args.map_scale > 1:
                    disp = cv2.resize(
                        disp,
                        (disp.shape[1] * args.map_scale, disp.shape[0] * args.map_scale),
                        interpolation=cv2.INTER_NEAREST,
                    )

                cv2.imshow("Unitree L2 Occupancy Map (2D)", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    occ_map.save(args.map_save_path)
                    print(f"Saved map: {args.map_save_path}")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            occ_map.save(args.map_save_path)
        except Exception:
            pass
        source.close()
        if not args.no_gui:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
