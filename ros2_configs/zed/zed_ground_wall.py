#!/usr/bin/env python3
"""
ZED 2i ground + wall segmentation (SDK Python).
This script classifies ground vs. non-ground points using a fitted plane.
It is safe to run without the camera connected (it will fail to open and exit).
"""

import sys
import time
import numpy as np

try:
    import pyzed.sl as sl
except Exception as exc:
    print("Failed to import pyzed.sl. Is the ZED SDK Python API installed?")
    print(f"Error: {exc}")
    sys.exit(1)

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def main():
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    zed = sl.Camera()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {err}")
        sys.exit(1)

    runtime = sl.RuntimeParameters()

    point_cloud = sl.Mat()
    image_left = sl.Mat()
    ground_plane = sl.Plane()
    tracking_reset = sl.Transform()

    if not HAS_CV2:
        print("OpenCV not found. Install it for live visualization:")
        print("  sudo apt install -y python3-opencv")

    print("Running. Press Ctrl+C to exit.")
    # Simple 2D occupancy map settings (XZ plane, Y up).
    # X: left/right, Z: forward. Units: meters.
    MAP_RES_M = 0.05
    MAP_WIDTH_M = 10.0
    MAP_HEIGHT_M = 10.0
    x_min = -MAP_WIDTH_M / 2.0
    x_max = MAP_WIDTH_M / 2.0
    z_min = 0.0
    z_max = MAP_HEIGHT_M

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # Retrieve point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            # Fit a ground plane using the SDK helper
            status = zed.find_floor_plane(ground_plane, tracking_reset)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"find_floor_plane failed: {status}")
                continue

            # Plane parameters: ax + by + cz + d = 0
            def plane_params(plane):
                # Support multiple ZED SDK Python API versions.
                if hasattr(plane, "normal"):
                    n = plane.normal
                    a0, b0, c0 = n.x, n.y, n.z
                    d0 = plane.distance if hasattr(plane, "distance") else plane.get_distance()
                    return a0, b0, c0, d0
                if hasattr(plane, "get_normal"):
                    n = plane.get_normal()
                    # n can be a struct with x/y/z or a sequence
                    if hasattr(n, "x"):
                        a0, b0, c0 = n.x, n.y, n.z
                    else:
                        a0, b0, c0 = n[0], n[1], n[2]
                    if hasattr(plane, "get_distance"):
                        d0 = plane.get_distance()
                    else:
                        eq = plane.get_plane_equation()
                        d0 = eq[3]
                    return a0, b0, c0, d0
                if hasattr(plane, "get_plane_equation"):
                    eq = plane.get_plane_equation()
                    return eq[0], eq[1], eq[2], eq[3]
                raise AttributeError("Unsupported Plane API: missing normal/normal getter")

            a, b, c, d = plane_params(ground_plane)

            # Sample a downscaled cloud to compute a quick summary
            cloud = point_cloud.get_data()
            if cloud is None:
                continue
            # Downsample for speed
            stride = 8
            xyz = cloud[::stride, ::stride, :3].reshape(-1, 3)
            # Filter invalid points
            mask = np.isfinite(xyz).all(axis=1)
            xyz = xyz[mask]
            if xyz.size == 0:
                continue

            # Distance to plane (signed)
            dist = (a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d) / np.sqrt(a * a + b * b + c * c)

            # Ground threshold: within 10 cm of plane
            ground_mask = np.abs(dist) < 0.10
            ground_pct = 100.0 * np.count_nonzero(ground_mask) / xyz.shape[0]

            # Wall/obstacle: above ground by > 10 cm
            obstacle_mask = dist > 0.10
            obstacle_pct = 100.0 * np.count_nonzero(obstacle_mask) / xyz.shape[0]

            print(f"Ground {ground_pct:5.1f}% | Obstacles {obstacle_pct:5.1f}% | Points {xyz.shape[0]}")

            # Build a simple 2D occupancy map from obstacle points.
            if HAS_CV2:
                map_vis = None
                obs = xyz[obstacle_mask]
                if obs.size > 0:
                    x = obs[:, 0]
                    z = obs[:, 2]
                    in_bounds = (x >= x_min) & (x < x_max) & (z >= z_min) & (z < z_max)
                    x = x[in_bounds]
                    z = z[in_bounds]
                    grid_w = int(MAP_WIDTH_M / MAP_RES_M)
                    grid_h = int(MAP_HEIGHT_M / MAP_RES_M)
                    occ = np.zeros((grid_h, grid_w), dtype=np.uint8)
                    ix = ((x - x_min) / MAP_RES_M).astype(np.int32)
                    iz = ((z - z_min) / MAP_RES_M).astype(np.int32)
                    # Flip Z so forward is "up" in the image.
                    occ[grid_h - 1 - iz, ix] = 255
                    map_vis = cv2.applyColorMap(occ, cv2.COLORMAP_BONE)
                else:
                    grid_w = int(MAP_WIDTH_M / MAP_RES_M)
                    grid_h = int(MAP_HEIGHT_M / MAP_RES_M)
                    map_vis = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            # Live visualization (optional)
            if HAS_CV2:
                img = image_left.get_data()
                if img is not None:
                    # ZED may return BGRA; drop alpha for overlay colors
                    if img.ndim == 3 and img.shape[2] == 4:
                        img = img[:, :, :3]
                    # Build a ground/obstacle mask at the same stride
                    xyz_full = cloud[::stride, ::stride, :3]
                    valid = np.isfinite(xyz_full).all(axis=2)
                    dist_full = (a * xyz_full[:, :, 0] + b * xyz_full[:, :, 1] + c * xyz_full[:, :, 2] + d)
                    denom = np.sqrt(a * a + b * b + c * c)
                    dist_full = dist_full / denom
                    dist_full[~valid] = np.nan

                    ground = (np.abs(dist_full) < 0.10) & valid
                    obstacle = (dist_full > 0.10) & valid

                    # Resize masks to full resolution
                    h, w, _ = img.shape
                    gh, gw = ground.shape
                    ground_full = cv2.resize(ground.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    obstacle_full = cv2.resize(obstacle.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

                    overlay = img.copy()
                    # Green for ground
                    overlay[ground_full == 1] = (0, 200, 0)
                    # Red for obstacles/walls
                    overlay[obstacle_full == 1] = (0, 0, 255)
                    # Blend
                    vis = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
                    cv2.imshow("ZED Ground/Obstacle Segmentation", vis)
                # Always show the map (even if the image frame is missing)
                if map_vis is not None:
                    cv2.imshow("ZED Occupancy Map (XZ)", map_vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            time.sleep(0.01)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
