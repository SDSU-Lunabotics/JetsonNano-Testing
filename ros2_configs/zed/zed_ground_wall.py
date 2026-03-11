#!/usr/bin/env python3
"""
ZED 2i ground + wall segmentation (SDK Python).
This script classifies ground vs. non-ground points using a fitted plane.
It is safe to run without the camera connected (it will fail to open and exit).
"""

import sys
import time
import argparse
import subprocess
import shutil
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

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header
    HAS_ROS2 = True
except Exception:
    HAS_ROS2 = False


def main():
    parser = argparse.ArgumentParser(description="ZED 2i ground + wall segmentation")
    parser.add_argument("--rviz", action="store_true", help="Launch rviz2 on startup")
    parser.add_argument("--rviz-config", default=None, help="Path to an RViz2 config file")
    parser.add_argument("--ros2", action="store_true", help="Publish a PointCloud2 topic over ROS2")
    parser.add_argument("--frame", default="zed_camera", help="Frame ID for ROS2 point cloud")
    args = parser.parse_args()

    if args.rviz:
        if shutil.which("rviz2") is None:
            print("rviz2 not found in PATH. Did you source ROS2?")
        else:
            rviz_config = args.rviz_config
            if rviz_config is None:
                rviz_config = os.path.join(os.path.dirname(__file__), "zed_pointcloud.rviz")
            if os.path.exists(rviz_config):
                print(f"Launching rviz2 with config: {rviz_config}")
                subprocess.Popen(["rviz2", "-d", rviz_config])
            else:
                print(f"RViz config not found: {rviz_config}. Launching default RViz.")
                subprocess.Popen(["rviz2"])

    node = None
    pc_pub = None
    pc_fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    if args.ros2:
        if not HAS_ROS2:
            print("ROS2 Python libs not found. Did you source ROS2 and install rclpy?")
        else:
            rclpy.init()
            node = rclpy.create_node("zed_ground_wall")
            pc_pub = node.create_publisher(PointCloud2, "zed/pointcloud", 10)
            print("ROS2 enabled: publishing /zed/pointcloud")
    else:
        print("ROS2 disabled (run with --ros2 to publish /zed/pointcloud)")

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
    MAP_DECAY = 0.97  # 1.0 = no decay, lower = faster fading
    x_min = -MAP_WIDTH_M / 2.0
    x_max = MAP_WIDTH_M / 2.0
    z_min = 0.0
    z_max = MAP_HEIGHT_M
    grid_w = int(MAP_WIDTH_M / MAP_RES_M)
    grid_h = int(MAP_HEIGHT_M / MAP_RES_M)
    map_counts = np.zeros((grid_h, grid_w), dtype=np.float32)

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
            # Ensure the plane normal points "up" (positive Y) so signed distance
            # is positive above the ground plane.
            if b < 0:
                a, b, c, d = -a, -b, -c, -d

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

            # Publish point cloud to ROS2 (optional)
            if pc_pub is not None:
                header = Header()
                header.stamp = node.get_clock().now().to_msg()
                header.frame_id = args.frame
                pts = xyz.astype(np.float32)
                msg = point_cloud2.create_cloud(header, pc_fields, pts.tolist())
                pc_pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.0)

            # Build a simple 2D top-down density map (XZ) from all valid points.
            if HAS_CV2:
                map_vis = None
                if xyz.size > 0:
                    x = xyz[:, 0]
                    z = xyz[:, 2]
                    in_bounds = (x >= x_min) & (x < x_max) & (z >= z_min) & (z < z_max)
                    x = x[in_bounds]
                    z = z[in_bounds]
                    counts = np.zeros((grid_h, grid_w), dtype=np.float32)
                    ix = ((x - x_min) / MAP_RES_M).astype(np.int32)
                    iz = ((z - z_min) / MAP_RES_M).astype(np.int32)
                    # Flip Z so forward is "up" in the image.
                    counts[grid_h - 1 - iz, ix] += 1.0
                    # Persistent map with decay.
                    map_counts *= MAP_DECAY
                    map_counts += counts
                    # Log-scale for visibility.
                    counts_f = np.log1p(map_counts)
                    if counts_f.max() > 0:
                        counts_f = counts_f / counts_f.max()
                    occ = (counts_f * 255.0).astype(np.uint8)
                    map_vis = cv2.applyColorMap(occ, cv2.COLORMAP_BONE)
                else:
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

    if node is not None:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
