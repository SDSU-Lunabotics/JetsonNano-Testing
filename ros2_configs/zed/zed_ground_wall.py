#!/usr/bin/env python3
"""
ZED 2i ground + wall segmentation (SDK Python).
This script classifies ground vs. non-ground points using a fitted plane.
It is safe to run without the camera connected (it will fail to open and exit).
"""

import sys
import time
import argparse
import os
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import rviz_utils
import ros2_utils
import segmentation
import map_utils
import zed_utils
import viewer_utils


def main():
    parser = argparse.ArgumentParser(description="ZED 2i ground + wall segmentation")
    parser.add_argument("--rviz", action="store_true", help="Launch rviz2 on startup")
    parser.add_argument("--rviz-config", default=None, help="Path to an RViz2 config file")
    parser.add_argument("--ros2", action="store_true", help="Publish a PointCloud2 topic over ROS2")
    parser.add_argument("--frame", default="zed_camera", help="Frame ID for ROS2 point cloud")
    parser.add_argument("--tracking", action="store_true", help="Enable ZED positional tracking")
    parser.add_argument("--map-width-m", type=float, default=20.0, help="Top-down map width in meters (X axis)")
    parser.add_argument("--map-height-m", type=float, default=20.0, help="Top-down map height in meters (Z axis)")
    parser.add_argument("--map-res-m", type=float, default=0.05, help="Map resolution in meters per cell")
    parser.add_argument("--map-z-min", type=float, default=0.0, help="Minimum Z (forward) bound for map")
    parser.add_argument("--map-scale", type=int, default=3, help="Upscale factor for map display window")
    parser.add_argument("--map-center", action="store_true", help="Center map on Z=0 (start camera in middle)")
    parser.add_argument("--map-save-path", default="zed_map.npz", help="Path to save persistent map data")
    parser.add_argument("--map-save-every", type=float, default=5.0, help="Seconds between map saves (0 to disable)")
    parser.add_argument("--map-load", action="store_true", help="Load existing map on startup if available")
    parser.add_argument("--map-decay", type=float, default=0.995, help="Map decay factor (1.0 = no decay)")
    parser.add_argument("--map-camera-size", type=int, default=3, help="Camera marker size in cells")
    parser.add_argument("--spatial-mapping", action="store_true", help="Enable ZED SDK spatial mapping")
    parser.add_argument("--spatial-res", default="medium", help="Spatial map resolution: low|medium|high")
    parser.add_argument("--spatial-range", default="medium", help="Spatial map range: short|medium|long")
    parser.add_argument("--spatial-save-path", default=None, help="Optional path to save spatial mesh (.obj)")
    parser.add_argument("--spatial-save-every", type=float, default=10.0, help="Seconds between spatial map saves")
    parser.add_argument("--spatial-viewer", action="store_true", help="Show live Open3D mesh viewer")
    args = parser.parse_args()

    if args.rviz_config is None:
        args.rviz_config = os.path.join(os.path.dirname(__file__), "zed_pointcloud.rviz")
    rviz_utils.launch_rviz(args.rviz, args.rviz_config)

    node, pc_pub, pc_fields = ros2_utils.setup_ros2(args.ros2)

    try:
        zed = zed_utils.open_zed_camera(sl)
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)

    runtime = sl.RuntimeParameters()

    point_cloud = sl.Mat()
    image_left = sl.Mat()
    ground_plane = sl.Plane()
    tracking_reset = sl.Transform()
    tracking_enabled = False
    pose_warned = False
    pose = None
    if args.tracking:
        tracking_enabled, pose = zed_utils.enable_tracking(zed, sl)
    spatial_enabled = False
    spatial_mesh = None
    last_spatial_save = time.time()
    if args.spatial_mapping:
        spatial_enabled, spatial_mesh = zed_utils.enable_spatial_mapping(
            zed,
            sl,
            resolution=args.spatial_res,
            mapping_range=args.spatial_range,
        )
    mesh_viewer = None
    if args.spatial_viewer:
        if not viewer_utils.HAS_OPEN3D:
            print("Open3D not available; mesh viewer disabled.")
        elif not args.spatial_save_path:
            print("Mesh viewer requires --spatial-save-path to load updates.")
        else:
            mesh_viewer = viewer_utils.MeshViewer()
            if not mesh_viewer.open():
                mesh_viewer = None

    if not HAS_CV2:
        print("OpenCV not found. Install it for live visualization:")
        print("  sudo apt install -y python3-opencv")

    print("Running. Press Ctrl+C to exit.")
    # Simple 2D occupancy map settings (XZ plane, Y up).
    # X: left/right, Z: forward. Units: meters.
    map_z_min = args.map_z_min
    if args.map_center:
        map_z_min = -args.map_height_m / 2.0

    occ_map = map_utils.OccupancyMap(
        map_res_m=args.map_res_m,
        map_width_m=args.map_width_m,
        map_height_m=args.map_height_m,
        map_z_min=map_z_min,
        decay=args.map_decay,
    )
    last_save = time.time()

    if args.map_load and os.path.exists(args.map_save_path):
        try:
            ok, msg = occ_map.load(args.map_save_path)
            print(f"{msg} ({args.map_save_path})" if ok else msg)
        except Exception as exc:
            print(f"Failed to load map ({args.map_save_path}): {exc}")

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # Update camera pose (world frame) if tracking is enabled.
            if tracking_enabled:
                R_world_cam, t_world_cam, pose_warned = zed_utils.get_world_transform(
                    zed, sl, pose, pose_warned
                )
            else:
                R_world_cam = np.eye(3, dtype=np.float32)
                t_world_cam = np.zeros(3, dtype=np.float32)

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

            a, b, c, d = segmentation.plane_params(ground_plane)
            a, b, c, d = segmentation.normalize_plane(a, b, c, d)

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
            dist, ground_mask, obstacle_mask = segmentation.classify_points(xyz, a, b, c, d, ground_thresh=0.10)
            ground_pct = 100.0 * np.count_nonzero(ground_mask) / xyz.shape[0]

            obstacle_pct = 100.0 * np.count_nonzero(obstacle_mask) / xyz.shape[0]

            print(f"Ground {ground_pct:5.1f}% | Obstacles {obstacle_pct:5.1f}% | Points {xyz.shape[0]}")

            # Publish point cloud to ROS2 (optional)
            ros2_utils.publish_pointcloud(node, pc_pub, pc_fields, xyz, args.frame)

            # Build a simple 2D top-down occupancy map (XZ) from ground/obstacle points.
            if HAS_CV2:
                map_vis = None
                if xyz.size > 0:
                    # Transform to world frame if tracking is enabled.
                    xyz_world = (R_world_cam @ xyz.T).T + t_world_cam
                    x = xyz_world[:, 0]
                    z = xyz_world[:, 2]
                    occ_map.update(x, z, ground_mask, obstacle_mask)
                    map_vis = occ_map.render()
                    # Draw camera position marker (blue square).
                    cam_row_col = occ_map.world_to_grid(float(t_world_cam[0]), float(t_world_cam[2]))
                    if cam_row_col is not None:
                        r0, c0 = cam_row_col
                        half = max(1, int(args.map_camera_size) // 2)
                        r1 = max(0, r0 - half)
                        r2 = min(occ_map.grid_h, r0 + half + 1)
                        c1 = max(0, c0 - half)
                        c2 = min(occ_map.grid_w, c0 + half + 1)
                        map_vis[r1:r2, c1:c2, :] = (255, 0, 0)
                        # Draw heading triangle (blue) if tracking is enabled.
                        if tracking_enabled and HAS_CV2:
                            # Camera forward axis in world frame (Z in camera frame).
                            forward = R_world_cam[:, 2]
                            fx, fz = float(forward[0]), float(forward[2])
                            ang = np.arctan2(fz, fx) + np.pi
                            size = max(3, int(args.map_camera_size) * 2)
                            tip_r = int(r0 - np.sin(ang) * size)
                            tip_c = int(c0 + np.cos(ang) * size)
                            left_ang = ang + 2.5
                            right_ang = ang - 2.5
                            base_r1 = int(r0 - np.sin(left_ang) * (size * 0.6))
                            base_c1 = int(c0 + np.cos(left_ang) * (size * 0.6))
                            base_r2 = int(r0 - np.sin(right_ang) * (size * 0.6))
                            base_c2 = int(c0 + np.cos(right_ang) * (size * 0.6))
                            tri = np.array(
                                [[tip_c, tip_r], [base_c1, base_r1], [base_c2, base_r2]],
                                dtype=np.int32,
                            )
                            cv2.fillConvexPoly(map_vis, tri, (255, 0, 0))
                    # Periodically save persistent map to disk.
                    if args.map_save_every > 0 and (time.time() - last_save) >= args.map_save_every:
                        occ_map.save(args.map_save_path)
                        last_save = time.time()
                    # Periodically update and save spatial map (mesh) if enabled.
                    if spatial_enabled and args.spatial_save_path and args.spatial_save_every > 0:
                        if (time.time() - last_spatial_save) >= args.spatial_save_every:
                            ok = zed_utils.update_spatial_map(zed, sl, spatial_mesh, args.spatial_save_path)
                            if ok:
                                last_spatial_save = time.time()
                                if mesh_viewer is not None:
                                    mesh_viewer.update_from_path(args.spatial_save_path)
                    if mesh_viewer is not None:
                        mesh_viewer.poll()
                else:
                    map_vis = np.zeros((occ_map.grid_h, occ_map.grid_w, 3), dtype=np.uint8)

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
                    if args.map_scale > 1:
                        map_vis = cv2.resize(
                            map_vis,
                            (occ_map.grid_w * args.map_scale, occ_map.grid_h * args.map_scale),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    cv2.imshow("ZED Occupancy Map (XZ)", map_vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            time.sleep(0.01)

    if spatial_enabled:
        zed_utils.disable_spatial_mapping(zed)
    if mesh_viewer is not None:
        mesh_viewer.close()
    ros2_utils.shutdown_ros2(node)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
