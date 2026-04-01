#!/usr/bin/env python3
"""
Standalone ZED zone datasheet logger for LiDAR comparison.

Logs random sampled map cells from obstacle (red) and hole (blue) zones
with mean world XYZ coordinates into a CSV file.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

try:
    import pyzed.sl as sl
except Exception as exc:
    print("Failed to import pyzed.sl. Is the ZED SDK Python API installed?")
    print(f"Error: {exc}")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import segmentation
import zed_utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Log random obstacle/hole ZED cells to CSV")
    p.add_argument("--out-csv", default=os.path.join(SCRIPT_DIR, "zone_datasheet.csv"), help="Output CSV path")
    p.add_argument("--tracking", action="store_true", help="Enable ZED positional tracking")
    p.add_argument("--stride", type=int, default=8, help="Point cloud downsample stride")
    p.add_argument("--sample-every-sec", type=float, default=1.0, help="Seconds between CSV sample writes")
    p.add_argument("--obstacle-samples", type=int, default=5, help="Random obstacle cells to log each cycle")
    p.add_argument("--hole-samples", type=int, default=3, help="Random hole cells to log each cycle")
    p.add_argument("--ground-samples", type=int, default=0, help="Random ground cells to log each cycle")
    p.add_argument("--obstacle-thresh-m", type=float, default=0.05, help="Obstacle height above floor (m)")
    p.add_argument("--hole-thresh-m", type=float, default=0.10, help="Hole depth below floor (m)")
    p.add_argument("--max-above-ground-m", type=float, default=1.22, help="Ignore points above this height over floor (m)")
    p.add_argument("--max-forward-m", type=float, default=6.0, help="Ignore points beyond this forward Z distance (m)")
    p.add_argument("--floor-update-sec", type=float, default=0.5, help="Seconds between floor plane updates")
    p.add_argument("--floor-min-normal-y", type=float, default=0.5, help="Reject floor plane if |normal.y| below this")
    p.add_argument("--map-width-m", type=float, default=20.0, help="Map width for cell indexing (X)")
    p.add_argument("--map-height-m", type=float, default=20.0, help="Map height for cell indexing (Z)")
    p.add_argument("--map-res-m", type=float, default=0.05, help="Map resolution (m/cell)")
    p.add_argument("--map-z-min", type=float, default=0.0, help="Map z min (ignored when --map-center)")
    p.add_argument("--map-center", action="store_true", help="Center map around z=0")
    p.add_argument("--seed", type=int, default=42, help="Random seed for cell sampling")
    return p.parse_args()


def ensure_csv_header(csv_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "unix_time",
                "iso_time",
                "frame_idx",
                "zone_type",
                "row",
                "col",
                "cell_world_x",
                "cell_world_z",
                "num_points_in_cell",
                "mean_x",
                "mean_y",
                "mean_z",
                "y_min",
                "y_max",
                "mean_dist_to_floor",
                "camera_x",
                "camera_y",
                "camera_z",
                "obstacle_thresh_m",
                "hole_thresh_m",
                "max_above_ground_m",
            ]
        )


def cell_center_world(row: int, col: int, x_min: float, z_min: float, map_res_m: float, grid_h: int) -> tuple[float, float]:
    wx = x_min + (float(col) + 0.5) * map_res_m
    wz = z_min + (float(grid_h - 1 - row) + 0.5) * map_res_m
    return wx, wz


def main() -> int:
    args = parse_args()
    ensure_csv_header(args.out_csv)

    map_z_min = -args.map_height_m / 2.0 if args.map_center else args.map_z_min
    x_min = -args.map_width_m / 2.0
    z_min = map_z_min
    grid_w = int(args.map_width_m / args.map_res_m)
    grid_h = int(args.map_height_m / args.map_res_m)

    rng = np.random.default_rng(int(args.seed))

    try:
        zed = zed_utils.open_zed_camera(sl)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    runtime = sl.RuntimeParameters()
    point_cloud = sl.Mat()
    ground_plane = sl.Plane()
    tracking_reset = sl.Transform()
    tracking_enabled = False
    pose_warned = False
    pose = None
    if args.tracking:
        tracking_enabled, pose = zed_utils.enable_tracking(zed, sl)

    has_plane = False
    a, b, c, d = 0.0, 1.0, 0.0, 0.0
    last_plane_update_time = 0.0
    frame_idx = 0
    last_sample_time = 0.0

    print(f"Logging zone datasheet to: {args.out_csv}")
    print("Press Ctrl+C to stop.")

    def write_samples(
        writer: csv.writer,
        zone_name: str,
        zone_mask: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        xyz_world: np.ndarray,
        dist_vals: np.ndarray,
        now_unix: float,
        camera_xyz: np.ndarray,
        max_samples: int,
    ) -> int:
        if max_samples <= 0:
            return 0
        if not np.any(zone_mask):
            return 0

        rc = np.stack((rows[zone_mask], cols[zone_mask]), axis=1)
        uniq_rc = np.unique(rc, axis=0)
        if uniq_rc.shape[0] == 0:
            return 0
        take_n = min(int(max_samples), int(uniq_rc.shape[0]))
        pick_idx = rng.choice(uniq_rc.shape[0], size=take_n, replace=False)
        picked = uniq_rc[pick_idx]

        wrote = 0
        iso = datetime.fromtimestamp(now_unix, tz=timezone.utc).isoformat()
        for r, c_ in picked:
            cell_sel = zone_mask & (rows == r) & (cols == c_)
            n_pts = int(np.count_nonzero(cell_sel))
            if n_pts <= 0:
                continue
            pts = xyz_world[cell_sel]
            dcell = dist_vals[cell_sel]
            mean_pt = np.mean(pts, axis=0)
            y_min = float(np.min(pts[:, 1]))
            y_max = float(np.max(pts[:, 1]))
            mean_dist = float(np.mean(dcell))
            cell_wx, cell_wz = cell_center_world(int(r), int(c_), x_min, z_min, args.map_res_m, grid_h)
            writer.writerow(
                [
                    f"{now_unix:.6f}",
                    iso,
                    frame_idx,
                    zone_name,
                    int(r),
                    int(c_),
                    f"{cell_wx:.6f}",
                    f"{cell_wz:.6f}",
                    n_pts,
                    f"{float(mean_pt[0]):.6f}",
                    f"{float(mean_pt[1]):.6f}",
                    f"{float(mean_pt[2]):.6f}",
                    f"{y_min:.6f}",
                    f"{y_max:.6f}",
                    f"{mean_dist:.6f}",
                    f"{float(camera_xyz[0]):.6f}",
                    f"{float(camera_xyz[1]):.6f}",
                    f"{float(camera_xyz[2]):.6f}",
                    f"{float(args.obstacle_thresh_m):.4f}",
                    f"{float(args.hole_thresh_m):.4f}",
                    f"{float(args.max_above_ground_m):.4f}",
                ]
            )
            wrote += 1
        return wrote

    try:
        with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            while True:
                if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    continue

                if tracking_enabled:
                    R_world_cam, t_world_cam, pose_warned = zed_utils.get_world_transform(
                        zed, sl, pose, pose_warned
                    )
                else:
                    R_world_cam = np.eye(3, dtype=np.float32)
                    t_world_cam = np.zeros(3, dtype=np.float32)

                now = time.time()
                should_update_plane = (not has_plane) or ((now - last_plane_update_time) >= args.floor_update_sec)
                if should_update_plane:
                    status = zed.find_floor_plane(ground_plane, tracking_reset)
                    last_plane_update_time = now
                    if status == sl.ERROR_CODE.SUCCESS:
                        a0, b0, c0, d0 = segmentation.plane_params(ground_plane)
                        a0, b0, c0, d0 = segmentation.normalize_plane(a0, b0, c0, d0)
                        if abs(float(b0)) >= float(args.floor_min_normal_y):
                            a, b, c, d = a0, b0, c0, d0
                            has_plane = True
                    if not has_plane:
                        continue

                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                cloud = point_cloud.get_data()
                if cloud is None:
                    continue

                stride = max(1, int(args.stride))
                xyz = cloud[::stride, ::stride, :3].reshape(-1, 3)
                valid = np.isfinite(xyz).all(axis=1)
                if args.max_forward_m > 0.0:
                    valid = valid & (xyz[:, 2] >= 0.0) & (xyz[:, 2] <= float(args.max_forward_m))
                xyz = xyz[valid]
                if xyz.size == 0:
                    continue

                dist, ground_mask, obstacle_mask = segmentation.classify_points(
                    xyz, a, b, c, d, ground_thresh=args.obstacle_thresh_m
                )
                if args.max_above_ground_m > 0.0:
                    keep = dist <= float(args.max_above_ground_m)
                    xyz = xyz[keep]
                    dist = dist[keep]
                    ground_mask = ground_mask[keep]
                    obstacle_mask = obstacle_mask[keep]
                    if xyz.size == 0:
                        continue
                hole_mask = dist < -args.hole_thresh_m

                xyz_world = (R_world_cam @ xyz.T).T + t_world_cam
                cols = ((xyz_world[:, 0] - x_min) / args.map_res_m).astype(np.int32)
                rows = (grid_h - 1 - ((xyz_world[:, 2] - z_min) / args.map_res_m)).astype(np.int32)
                inb = (
                    (rows >= 0)
                    & (rows < grid_h)
                    & (cols >= 0)
                    & (cols < grid_w)
                )
                if not np.any(inb):
                    continue

                rows = rows[inb]
                cols = cols[inb]
                xyz_world = xyz_world[inb]
                dist = dist[inb]
                ground_mask = ground_mask[inb]
                obstacle_mask = obstacle_mask[inb]
                hole_mask = hole_mask[inb]

                if (now - last_sample_time) < max(0.05, float(args.sample_every_sec)):
                    frame_idx += 1
                    continue
                last_sample_time = now

                wrote_obs = write_samples(
                    writer=writer,
                    zone_name="obstacle",
                    zone_mask=obstacle_mask,
                    rows=rows,
                    cols=cols,
                    xyz_world=xyz_world,
                    dist_vals=dist,
                    now_unix=now,
                    camera_xyz=t_world_cam,
                    max_samples=args.obstacle_samples,
                )
                wrote_hole = write_samples(
                    writer=writer,
                    zone_name="hole",
                    zone_mask=hole_mask,
                    rows=rows,
                    cols=cols,
                    xyz_world=xyz_world,
                    dist_vals=dist,
                    now_unix=now,
                    camera_xyz=t_world_cam,
                    max_samples=args.hole_samples,
                )
                wrote_ground = write_samples(
                    writer=writer,
                    zone_name="ground",
                    zone_mask=ground_mask,
                    rows=rows,
                    cols=cols,
                    xyz_world=xyz_world,
                    dist_vals=dist,
                    now_unix=now,
                    camera_xyz=t_world_cam,
                    max_samples=args.ground_samples,
                )
                f.flush()
                print(
                    f"frame={frame_idx} logged rows -> obstacle={wrote_obs} hole={wrote_hole} ground={wrote_ground}"
                )
                frame_idx += 1
    except KeyboardInterrupt:
        print("\nStopped zone datasheet logger.")
    finally:
        try:
            zed.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

