#!/usr/bin/env python3
"""
Simple ground/wall terrain viewer for ZED in perception-testing.

Purpose:
- Keep setup minimal and fast.
- Show camera overlay (ground/obstacle/hole) and a top-down occupancy map.
- Reuse ZEDAuto helpers so behavior is familiar.
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import cv2
except Exception as exc:
    print("OpenCV is required.")
    print(f"Error: {exc}")
    sys.exit(1)

try:
    import pyzed.sl as sl
except Exception as exc:
    print("Failed to import pyzed.sl. Install ZED SDK Python API.")
    print(f"Error: {exc}")
    sys.exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
ZEDAUTO_DIR = os.path.join(REPO_ROOT, "ZEDAuto")
if ZEDAUTO_DIR not in sys.path:
    sys.path.insert(0, ZEDAUTO_DIR)

import segmentation
import map_utils
import zed_utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple ZED ground/wall terrain viewer")
    p.add_argument("--tracking", action="store_true", help="Enable positional tracking for world-fixed map")
    p.add_argument("--floor-update-sec", type=float, default=0.6, help="Seconds between floor plane updates")
    p.add_argument("--floor-min-normal-y", type=float, default=0.55, help="Reject floor planes with |normal.y| below this")
    p.add_argument("--stride", type=int, default=8, help="Point cloud sampling stride")
    p.add_argument("--obstacle-thresh-m", type=float, default=0.06, help="Obstacle height above floor (m)")
    p.add_argument("--hole-thresh-m", type=float, default=0.08, help="Hole depth below floor (m)")
    p.add_argument("--max-above-ground-m", type=float, default=1.0, help="Ignore points above this floor-relative height (m)")
    p.add_argument("--max-forward-m", type=float, default=6.0, help="Ignore points farther than this forward distance (m)")
    p.add_argument("--show-holes", action="store_true", help="Show holes (blue) in overlay and map")
    p.add_argument("--map-width-m", type=float, default=20.0, help="Map width in meters")
    p.add_argument("--map-height-m", type=float, default=20.0, help="Map height in meters")
    p.add_argument("--map-res-m", type=float, default=0.05, help="Map resolution in meters/cell")
    p.add_argument("--map-z-min", type=float, default=0.0, help="Map minimum forward bound")
    p.add_argument("--map-center", action="store_true", help="Center map around start position")
    p.add_argument("--map-scale", type=int, default=3, help="Display scale for map window")
    p.add_argument("--map-decay", type=float, default=0.995, help="Base map decay")
    p.add_argument("--free-decay", type=float, default=1.0, help="Free-space decay")
    p.add_argument("--occ-decay", type=float, default=1.0, help="Obstacle decay")
    p.add_argument("--hole-decay", type=float, default=1.0, help="Hole decay")
    return p.parse_args()


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] > 3:
            return img[:, :, :3]
    return img


def main() -> None:
    args = parse_args()

    zed = zed_utils.open_zed_camera(sl)
    runtime = sl.RuntimeParameters()
    point_cloud = sl.Mat()
    image_left = sl.Mat()
    ground_plane = sl.Plane()
    tracking_reset = sl.Transform()

    tracking_enabled = False
    tracking_pose_ok = True
    pose_warned = False
    pose = None
    if args.tracking:
        tracking_enabled, pose = zed_utils.enable_tracking(zed, sl)
        tracking_pose_ok = not tracking_enabled
    last_valid_R = np.eye(3, dtype=np.float32)
    last_valid_t = np.zeros(3, dtype=np.float32)

    map_z_min = -args.map_height_m / 2.0 if args.map_center else args.map_z_min
    occ_map = map_utils.OccupancyMap(
        map_res_m=args.map_res_m,
        map_width_m=args.map_width_m,
        map_height_m=args.map_height_m,
        map_z_min=map_z_min,
        decay=args.map_decay,
        free_decay=args.free_decay,
        occ_decay=args.occ_decay,
        hole_decay=args.hole_decay,
    )

    has_plane = False
    last_plane_update = 0.0
    a, b, c, d = 0.0, 1.0, 0.0, 0.0
    last_print = 0.0

    print("Simple terrain view running. Keys: q/Esc quit, r refresh floor, c clear map.")

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            if tracking_enabled:
                R, t, pose_warned, tracking_pose_ok = zed_utils.get_world_transform_with_status(
                    zed, sl, pose, pose_warned
                )
                if tracking_pose_ok:
                    last_valid_R = R
                    last_valid_t = t
                else:
                    R = last_valid_R
                    t = last_valid_t
            else:
                R = np.eye(3, dtype=np.float32)
                t = np.zeros(3, dtype=np.float32)

            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            cloud = point_cloud.get_data()
            img = _to_bgr(image_left.get_data())

            now = time.time()
            if (not has_plane) or (now - last_plane_update >= max(0.05, float(args.floor_update_sec))):
                status = zed.find_floor_plane(ground_plane, tracking_reset)
                last_plane_update = now
                if status == sl.ERROR_CODE.SUCCESS:
                    a0, b0, c0, d0 = segmentation.plane_params(ground_plane)
                    a0, b0, c0, d0 = segmentation.canonical_plane(a0, b0, c0, d0)
                    if abs(float(b0)) >= float(args.floor_min_normal_y):
                        a, b, c, d = a0, b0, c0, d0
                        has_plane = True

            if cloud is None:
                continue

            stride = max(1, int(args.stride))
            xyz = cloud[::stride, ::stride, :3].reshape(-1, 3)
            mask = np.isfinite(xyz).all(axis=1)
            if float(args.max_forward_m) > 0.0:
                mask &= (xyz[:, 2] >= 0.0) & (xyz[:, 2] <= float(args.max_forward_m))
            xyz = xyz[mask]

            if has_plane and xyz.size > 0:
                denom = np.sqrt(a * a + b * b + c * c)
                dist = (a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d) / denom
                obstacle_mask = dist > float(args.obstacle_thresh_m)
                ground_mask = (dist >= -float(args.hole_thresh_m)) & (dist <= float(args.obstacle_thresh_m))
                if float(args.max_above_ground_m) > 0.0:
                    above_ok = dist <= float(args.max_above_ground_m)
                    obstacle_mask &= above_ok
                    ground_mask &= above_ok
                if args.show_holes:
                    hole_mask = dist < -float(args.hole_thresh_m)
                else:
                    hole_mask = np.zeros(dist.shape, dtype=bool)

                # Keep map integration alive using the best available pose (current or held last-valid).
                xyz_world = (R @ xyz.T).T + t
                occ_map.update(
                    x=xyz_world[:, 0],
                    z=xyz_world[:, 2],
                    ground_mask=ground_mask,
                    obstacle_mask=obstacle_mask,
                    hole_mask=hole_mask,
                )

                if now - last_print >= 1.0:
                    n = max(1, xyz.shape[0])
                    gp = 100.0 * np.count_nonzero(ground_mask) / n
                    op = 100.0 * np.count_nonzero(obstacle_mask) / n
                    hp = 100.0 * np.count_nonzero(hole_mask) / n
                    print(f"Ground {gp:5.1f}% | Obstacles {op:5.1f}% | Holes {hp:5.1f}% | Points {n}")
                    last_print = now

            # Camera overlay
            if img is not None and has_plane:
                xyz_small = cloud[::stride, ::stride, :3]
                valid = np.isfinite(xyz_small).all(axis=2)
                denom = np.sqrt(a * a + b * b + c * c)
                dist_num = (a * xyz_small[:, :, 0] + b * xyz_small[:, :, 1] + c * xyz_small[:, :, 2] + d)
                dist_small = np.full_like(dist_num, np.nan, dtype=np.float32)
                if np.any(valid):
                    dist_small[valid] = (dist_num[valid] / denom).astype(np.float32)

                obstacle_small = (dist_small > float(args.obstacle_thresh_m)) & valid
                ground_small = (
                    (dist_small >= -float(args.hole_thresh_m))
                    & (dist_small <= float(args.obstacle_thresh_m))
                    & valid
                )
                if float(args.max_above_ground_m) > 0.0:
                    above_ok = dist_small <= float(args.max_above_ground_m)
                    obstacle_small &= above_ok
                    ground_small &= above_ok
                if args.show_holes:
                    hole_small = (dist_small < -float(args.hole_thresh_m)) & valid
                else:
                    hole_small = np.zeros_like(valid, dtype=bool)

                h, w, _ = img.shape
                ground = cv2.resize(ground_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                obstacle = cv2.resize(obstacle_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                hole = cv2.resize(hole_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

                overlay = img.copy()
                overlay[ground == 1] = (0, 200, 0)
                overlay[obstacle == 1] = (0, 0, 255)
                if args.show_holes:
                    overlay[hole == 1] = (255, 70, 0)
                vis = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
                cv2.imshow("Terrain Camera (Ground/Wall/Hole)", vis)

            # Top-down map
            map_vis = occ_map.render()
            cam_rc = occ_map.world_to_grid(float(t[0]), float(t[2]))
            if cam_rc is not None:
                rr, cc = cam_rc
                r1, r2 = max(0, rr - 1), min(occ_map.grid_h, rr + 2)
                c1, c2 = max(0, cc - 1), min(occ_map.grid_w, cc + 2)
                map_vis[r1:r2, c1:c2] = (255, 255, 255)
            scale = max(1, int(args.map_scale))
            if scale > 1:
                map_vis = cv2.resize(
                    map_vis,
                    (occ_map.grid_w * scale, occ_map.grid_h * scale),
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow("Terrain Map (XZ)", map_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                has_plane = False
                print("Floor plane refresh requested.")
            if key == ord("c"):
                occ_map.free_counts.fill(0.0)
                occ_map.occ_counts.fill(0.0)
                occ_map.hole_counts.fill(0.0)
                print("Map cleared.")
    finally:
        try:
            zed.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

