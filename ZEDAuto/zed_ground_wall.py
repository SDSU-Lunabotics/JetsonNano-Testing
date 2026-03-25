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
import math
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
import heatmap_utils
import zed_utils
import viewer_utils
import stream_utils

try:
    from networktables import NetworkTables
    HAS_NT = True
except Exception:
    HAS_NT = False


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
    parser.add_argument("--free-decay", type=float, default=None, help="Free-space decay (defaults to --map-decay)")
    parser.add_argument("--free-decay-unconfirmed", type=float, default=None, help="Decay for low-confidence free cells")
    parser.add_argument("--free-decay-confirmed", type=float, default=1.0, help="Decay for confirmed free cells")
    parser.add_argument("--free-confirm-hits", type=float, default=8.0, help="Hits needed to mark a free cell as confirmed")
    parser.add_argument("--free-confirm-ratio", type=float, default=1.2, help="Free/obstacle ratio required for confirmed free")
    parser.add_argument("--free-downgrade-factor", type=float, default=0.6, help="Multiply free confidence when obstacle/hole appears")
    parser.add_argument("--occ-decay", type=float, default=None, help="Obstacle decay (defaults to --map-decay)")
    parser.add_argument("--hole-decay", type=float, default=None, help="Hole decay (defaults to --map-decay)")
    parser.add_argument("--map-camera-size", type=int, default=3, help="Camera marker size in cells")
    parser.add_argument("--heatmap", action="store_true", help="Enable heatmap view for map confidence/risk")
    parser.add_argument(
        "--heatmap-mode",
        default="risk",
        choices=["risk", "obstacle", "hole", "free", "evidence"],
        help="Heatmap mode",
    )
    parser.add_argument("--heatmap-alpha", type=float, default=0.35, help="Heatmap overlay alpha (0-1)")
    parser.add_argument(
        "--heatmap-min-evidence",
        type=float,
        default=1.0,
        help="Min evidence needed before a heatmap cell is shown",
    )
    parser.add_argument(
        "--heatmap-window",
        action="store_true",
        help="Show heatmap in a separate window instead of overlaying on map",
    )
    parser.add_argument("--obstacle-thresh-m", type=float, default=0.05, help="Obstacle height above ground (m)")
    parser.add_argument("--hole-thresh-m", type=float, default=0.05, help="Hole depth below ground (m)")
    parser.add_argument("--disable-holes", action="store_true", help="Disable hole detection (testing)")
    parser.add_argument("--path-avoid-occ-min", type=float, default=3.0, help="Min obstacle count for path blocking")
    parser.add_argument("--path-avoid-occ-ratio", type=float, default=1.5, help="Min occupied/free ratio for blocking")
    parser.add_argument("--path-connectivity", type=int, default=8, choices=[4, 8], help="A* grid connectivity")
    parser.add_argument("--path-replan-sec", type=float, default=0.5, help="How often to retry path planning")
    parser.add_argument("--block-unknown", action="store_true", help="Treat unknown (black) cells as blocked")
    parser.add_argument("--unknown-min-evidence", type=float, default=1.0, help="Evidence threshold to mark a cell as known")
    parser.add_argument("--start-clear-radius-m", type=float, default=0.35, help="Clear blocked cells near rover start/blind spot")
    parser.add_argument("--rover-size-m", type=float, default=0.305, help="Rover footprint size (m, square)")
    parser.add_argument("--spatial-mapping", action="store_true", help="Enable ZED SDK spatial mapping")
    parser.add_argument("--spatial-res", default="medium", help="Spatial map resolution: low|medium|high")
    parser.add_argument("--spatial-range", default="medium", help="Spatial map range: short|medium|long")
    parser.add_argument("--spatial-save-path", default=None, help="Optional path to save spatial mesh (.obj)")
    parser.add_argument("--spatial-save-every", type=float, default=10.0, help="Seconds between spatial map saves")
    parser.add_argument("--spatial-viewer", action="store_true", help="Show live Open3D mesh viewer")
    parser.add_argument("--spatial-filter", default="none", help="Mesh filter: none|low|medium|high")
    parser.add_argument("--drive", action="store_true", help="Enable RoboRIO driving commands")
    parser.add_argument("--roborio-ip", default="10.0.9.2", help="RoboRIO IP for NetworkTables")
    parser.add_argument("--drive-speed", type=float, default=0.7, help="Forward speed command (0-1)")
    parser.add_argument("--drive-turn-k", type=float, default=0.8, help="Turn gain for heading error")
    parser.add_argument("--drive-rate-hz", type=float, default=10.0, help="Drive command rate (Hz)")
    parser.add_argument("--drive-goal-tol-m", type=float, default=0.3, help="Goal tolerance (m)")
    parser.add_argument("--drive-heading-tol-deg", type=float, default=10.0, help="Heading tolerance (deg)")
    parser.add_argument("--drive-heading-flip", action="store_true", help="Flip heading by 180 degrees")
    parser.add_argument(
        "--drive-ready-pulse-sec",
        type=float,
        default=0.10,
        help="How long CommandReady stays high per command pulse",
    )
    parser.add_argument("--drive-debug", action="store_true", help="Print outgoing NT drive commands")
    parser.add_argument(
        "--nt-health-debug",
        action="store_true",
        help="Print NetworkTables session health and robot-published Jetson drive keys",
    )
    parser.add_argument(
        "--nt-health-period-sec",
        type=float,
        default=1.0,
        help="Seconds between NetworkTables health debug prints",
    )
    parser.add_argument(
        "--nt-enable-heartbeat-sec",
        type=float,
        default=0.10,
        help="Seconds between automation-state heartbeat writes while driving",
    )
    parser.add_argument(
        "--nt-command-ack-timeout-sec",
        type=float,
        default=0.30,
        help="Seconds to wait before clearing a stuck CommandReady flag",
    )
    parser.add_argument(
        "--nt-forward-scale",
        type=float,
        default=1.0,
        help="Value written to Jetson/Speed while automation is enabled",
    )
    parser.add_argument(
        "--nt-turn-scale",
        type=float,
        default=1.0,
        help="Value written to Jetson/TurnSpeed while automation is enabled",
    )
    parser.add_argument("--floor-update-sec", type=float, default=0.5, help="Seconds between floor-plane updates")
    parser.add_argument("--floor-min-normal-y", type=float, default=0.5, help="Reject floor planes with |normal.y| below this")
    parser.add_argument("--stream-ip", default=None, help="UDP target IP for GStreamer stream")
    parser.add_argument("--stream-port", type=int, default=5600, help="UDP port for GStreamer stream")
    parser.add_argument("--stream-fps", type=float, default=15.0, help="Stream FPS")
    parser.add_argument("--stream-bitrate-kbps", type=int, default=2500, help="Stream bitrate in kbps")
    parser.add_argument("--stream-view", default="both", choices=["camera", "map", "both"], help="Which view to stream")
    parser.add_argument("--no-gui", action="store_true", help="Disable local OpenCV windows")
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
        free_decay=args.free_decay,
        free_decay_unconfirmed=args.free_decay_unconfirmed,
        free_decay_confirmed=args.free_decay_confirmed,
        free_confirm_hits=args.free_confirm_hits,
        free_confirm_ratio=args.free_confirm_ratio,
        free_downgrade_factor=args.free_downgrade_factor,
        occ_decay=args.occ_decay,
        hole_decay=args.hole_decay,
    )
    last_save = time.time()

    if args.map_load and os.path.exists(args.map_save_path):
        try:
            ok, msg = occ_map.load(args.map_save_path)
            print(f"{msg} ({args.map_save_path})" if ok else msg)
        except Exception as exc:
            print(f"Failed to load map ({args.map_save_path}): {exc}")

    sd = None
    if args.drive:
        if not HAS_NT:
            print("NetworkTables not available; disable --drive or install pynetworktables.")
        else:
            NetworkTables.initialize(server=args.roborio_ip)
            sd = NetworkTables.getTable("SmartDashboard")
            print(f"Drive enabled: NetworkTables to {args.roborio_ip}")

    goal_cell = None
    path_cells = None
    last_path_cells = None
    last_start = None
    last_goal = None
    map_window_ready = False
    emergency_stop = False
    last_drive_send = 0.0
    manual_fwd = 0.0
    manual_turn = 0.0
    manual_mode = False
    last_w_time = 0.0
    last_s_time = 0.0
    last_a_time = 0.0
    last_d_time = 0.0
    key_hold_timeout = 0.35
    nt_last_conn_log = 0.0
    nt_last_health_log = 0.0
    nt_health_seq = 0
    nt_command_seq = 0
    nt_ready_stuck_since = 0.0
    nt_last_auto_push = 0.0
    nt_ready_high = False
    nt_ready_clear_time = 0.0
    last_drive_debug_time = 0.0
    nt_connected_cached = False
    status_cmd_enabled = False
    status_cmd_fwd = 0.0
    status_cmd_turn = 0.0
    status_cmd_duration = 0.0
    status_target_cell = None
    status_target_world = None
    last_path_plan_time = 0.0
    last_plane_update_time = 0.0
    plane_fail_count = 0
    a, b, c, d = 0.0, 1.0, 0.0, 0.0
    has_plane = False

    def on_map_click(event, x, y, flags, param):
        nonlocal goal_cell, path_cells, last_path_cells, last_start, last_goal
        nonlocal emergency_stop, last_path_plan_time
        if event != cv2.EVENT_LBUTTONDOWN:
            if event == cv2.EVENT_RBUTTONDOWN:
                emergency_stop = True
                print("EMERGENCY STOP")
            return
        scale = max(1, int(args.map_scale))
        row = int(y / scale)
        col = int(x / scale)
        if row < 0 or row >= occ_map.grid_h or col < 0 or col >= occ_map.grid_w:
            return
        goal_cell = (row, col)
        path_cells = None
        last_path_cells = None
        last_start = None
        last_goal = None
        last_path_plan_time = 0.0
        emergency_stop = False
        print(f"New goal set at row={row}, col={col}")

    def send_nt_command(enabled, fwd, turn, duration):
        nonlocal nt_command_seq, nt_ready_stuck_since, nt_last_auto_push
        nonlocal nt_ready_high, nt_ready_clear_time, last_drive_debug_time
        nonlocal status_cmd_enabled, status_cmd_fwd, status_cmd_turn, status_cmd_duration
        if sd is None:
            return
        now = time.time()
        enabled = bool(enabled)
        status_cmd_enabled = bool(enabled)
        status_cmd_fwd = float(fwd)
        status_cmd_turn = float(turn)
        status_cmd_duration = float(duration)

        def push_automation_state(force=False):
            nonlocal nt_last_auto_push
            if (not force) and (now - nt_last_auto_push) < max(0.02, float(args.nt_enable_heartbeat_sec)):
                return
            sd.putBoolean("Jetson/AutomationEnabled", enabled)
            # Robot-side code may scale command by these keys.
            if enabled:
                sd.putNumber("Jetson/Speed", float(args.nt_forward_scale))
                sd.putNumber("Jetson/TurnSpeed", float(args.nt_turn_scale))
            else:
                sd.putNumber("Jetson/Speed", 0.0)
                sd.putNumber("Jetson/TurnSpeed", 0.0)
            nt_last_auto_push = now

        push_automation_state(force=not enabled)
        if not enabled:
            # Clear string-based legacy command channels when auto-drive is off.
            sd.putString("Jetson/Command", "")
            sd.putBoolean("Jetson/CommandReady", False)
            nt_ready_high = False
            nt_ready_stuck_since = 0.0
            return

        # Avoid stomping in-flight commands if robot has not consumed CommandReady yet.
        remote_ready = sd.getBoolean("Jetson/CommandReady", False)
        if remote_ready and not nt_ready_high:
            if nt_ready_stuck_since <= 0.0:
                nt_ready_stuck_since = now
            if (now - nt_ready_stuck_since) >= max(0.05, float(args.nt_command_ack_timeout_sec)):
                if args.drive_debug:
                    print("Warning: stale CommandReady detected; clearing flag.")
                sd.putBoolean("Jetson/CommandReady", False)
                nt_ready_stuck_since = 0.0
            else:
                return
        else:
            nt_ready_stuck_since = 0.0

        nt_command_seq += 1
        sd.putNumber("Jetson/CommandForward", float(fwd))
        sd.putNumber("Jetson/CommandTurn", float(turn))
        sd.putNumber("Jetson/CommandDuration", float(duration))
        sd.putNumber("Jetson/CommandSeq", float(nt_command_seq))

        # Pulse CommandReady high, then clear shortly after.
        # Keep it near configured value, but ensure a short low interval exists each cycle.
        cmd_period = max(0.02, float(duration))
        pulse_sec = max(0.01, float(args.drive_ready_pulse_sec))
        pulse_sec = min(pulse_sec, max(0.01, cmd_period - 0.01))
        if not nt_ready_high:
            sd.putBoolean("Jetson/CommandReady", True)
            nt_ready_high = True
            nt_ready_clear_time = now + pulse_sec
        if args.drive_debug:
            if (now - last_drive_debug_time) >= 0.2:
                print(
                    f"NT cmd seq={nt_command_seq:.0f} enabled={bool(enabled)} "
                    f"fwd={float(fwd):+.2f} turn={float(turn):+.2f} "
                    f"dur={float(duration):.2f} pulse={pulse_sec:.2f}"
                )
                last_drive_debug_time = now

    def nt_connections_summary():
        if not HAS_NT:
            return "nt-disabled"
        try:
            conns = NetworkTables.getConnections()
        except Exception:
            return "unavailable"
        if not conns:
            return "none"
        parts = []
        for conn in conns[:3]:
            remote_id = getattr(conn, "remote_id", "?")
            remote_ip = getattr(conn, "remote_ip", "?")
            parts.append(f"{remote_id}@{remote_ip}")
        if len(conns) > 3:
            parts.append(f"+{len(conns) - 3} more")
        return ", ".join(parts)

    def render_status_panel(cam_cell):
        panel_h = 300
        panel_w = 620
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (24, 24, 24)

        def put_line(text, y, color=(235, 235, 235), scale=0.55):
            cv2.putText(
                panel,
                text,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                1,
                cv2.LINE_AA,
            )

        def draw_axis(label, value, y):
            value = max(-1.0, min(1.0, float(value)))
            put_line(f"{label}: {value:+.2f}", y - 10)
            x0, x1 = 170, panel_w - 20
            cx = (x0 + x1) // 2
            cv2.line(panel, (x0, y), (x1, y), (90, 90, 90), 1)
            cv2.line(panel, (cx, y - 9), (cx, y + 9), (140, 140, 140), 1)
            half = (x1 - x0) // 2
            vx = int(cx + value * half)
            color = (0, 220, 0) if abs(value) <= 0.05 else ((0, 220, 255) if value > 0 else (255, 180, 0))
            cv2.rectangle(panel, (min(cx, vx), y - 7), (max(cx, vx), y + 7), color, -1)

        if not args.drive:
            mode_label = "DRIVE OFF"
            mode_color = (180, 180, 180)
        elif emergency_stop:
            mode_label = "STOPPED"
            mode_color = (0, 80, 255)
        elif manual_mode:
            mode_label = "MANUAL"
            mode_color = (0, 220, 255)
        elif goal_cell is not None:
            mode_label = "AUTO"
            mode_color = (0, 220, 0)
        else:
            mode_label = "IDLE"
            mode_color = (180, 180, 180)

        put_line("ZED DRIVE STATUS", 30, (255, 255, 255), 0.72)
        put_line(f"Mode: {mode_label}", 62, mode_color, 0.62)
        put_line(f"E-stop: {'ON' if emergency_stop else 'OFF'}", 88, (0, 80, 255) if emergency_stop else (170, 255, 170))
        put_line(f"NT connected: {nt_connected_cached}", 114, (170, 255, 170) if nt_connected_cached else (140, 140, 255))

        if goal_cell is None:
            put_line("Goal cell: none", 148, (190, 190, 190))
            put_line("Goal world: none", 172, (190, 190, 190))
        else:
            goal_world = occ_map.grid_to_world(goal_cell[0], goal_cell[1])
            put_line(f"Goal cell: r={goal_cell[0]} c={goal_cell[1]}", 148, (220, 240, 255))
            if goal_world is None:
                put_line("Goal world: unavailable", 172, (190, 190, 190))
            else:
                put_line(f"Goal world: x={goal_world[0]:+.2f} z={goal_world[1]:+.2f}", 172, (220, 240, 255))

        if status_target_world is None:
            put_line("Active target: none", 196, (190, 190, 190))
        else:
            tc = status_target_cell
            if tc is None:
                put_line(
                    f"Active target: x={status_target_world[0]:+.2f} z={status_target_world[1]:+.2f}",
                    196,
                    (255, 235, 170),
                )
            else:
                put_line(
                    f"Active target: r={tc[0]} c={tc[1]} x={status_target_world[0]:+.2f} z={status_target_world[1]:+.2f}",
                    196,
                    (255, 235, 170),
                )

        if cam_cell is None:
            put_line("Robot cell: unavailable", 220, (190, 190, 190))
        else:
            put_line(f"Robot cell: r={cam_cell[0]} c={cam_cell[1]}", 220, (180, 255, 220))

        put_line(
            f"Last command: {'ENABLED' if status_cmd_enabled else 'DISABLED'} dur={status_cmd_duration:.2f}s",
            246,
            (190, 255, 190) if status_cmd_enabled else (190, 190, 190),
        )
        draw_axis("Forward", status_cmd_fwd, 270)
        draw_axis("Turn", status_cmd_turn, 292)
        return panel

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
            # Fit/update floor plane on an interval to reduce SDK load.
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
                        plane_fail_count = 0
                    else:
                        plane_fail_count += 1
                        if plane_fail_count % 10 == 1:
                            print(f"Rejected floor plane with weak normal.y={b0:.3f}")
                else:
                    plane_fail_count += 1
                    if plane_fail_count % 10 == 1:
                        print(f"find_floor_plane failed: {status}")

            if not has_plane:
                # No valid plane yet; wait for a stable floor estimate.
                continue

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
            dist, ground_mask, obstacle_mask = segmentation.classify_points(
                xyz, a, b, c, d, ground_thresh=args.obstacle_thresh_m
            )
            if args.disable_holes:
                hole_mask = np.zeros(dist.shape, dtype=bool)
            else:
                hole_mask = dist < -args.hole_thresh_m
            ground_pct = 100.0 * np.count_nonzero(ground_mask) / xyz.shape[0]

            obstacle_pct = 100.0 * np.count_nonzero(obstacle_mask) / xyz.shape[0]
            hole_pct = 100.0 * np.count_nonzero(hole_mask) / xyz.shape[0]

            print(
                f"Ground {ground_pct:5.1f}% | Obstacles {obstacle_pct:5.1f}% "
                f"| Holes {hole_pct:5.1f}% | Points {xyz.shape[0]}"
            )

            # Publish point cloud to ROS2 (optional)
            ros2_utils.publish_pointcloud(node, pc_pub, pc_fields, xyz, args.frame)

            # Build a simple 2D top-down occupancy map (XZ) from ground/obstacle points.
            if HAS_CV2:
                map_vis = None
                heatmap_vis = None
                cam_row_col = None
                if xyz.size > 0:
                    # Transform to world frame if tracking is enabled.
                    xyz_world = (R_world_cam @ xyz.T).T + t_world_cam
                    x = xyz_world[:, 0]
                    z = xyz_world[:, 2]
                    occ_map.update(x, z, ground_mask, obstacle_mask, hole_mask)
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

                    # Compute/update path to goal (avoid red obstacles only).
                    if goal_cell is not None and cam_row_col is not None:
                        now = time.time()
                        should_replan = (
                            cam_row_col != last_start
                            or goal_cell != last_goal
                            or path_cells is None
                            or (now - last_path_plan_time) >= args.path_replan_sec
                        )
                        if should_replan:
                            obs = occ_map.obstacle_mask(
                                min_occ_count=args.path_avoid_occ_min,
                                min_occ_ratio=args.path_avoid_occ_ratio,
                            )
                            if args.block_unknown:
                                known = occ_map.known_mask(min_evidence=args.unknown_min_evidence)
                                obs = np.logical_or(obs, np.logical_not(known))
                            radius_cells = int(np.ceil((args.rover_size_m / 2.0) / occ_map.map_res_m))
                            if radius_cells > 0:
                                obs = map_utils.inflate_mask(obs, radius_cells)
                            clear_cells = int(np.ceil(max(0.0, args.start_clear_radius_m) / occ_map.map_res_m))
                            if clear_cells > 0:
                                obs = map_utils.clear_mask_circle(obs, cam_row_col, clear_cells)
                            path_cells = map_utils.astar_path(
                                cam_row_col,
                                goal_cell,
                                obs,
                                connectivity=args.path_connectivity,
                            )
                            if path_cells:
                                last_path_cells = path_cells
                            else:
                                # Do not keep stale path to an old goal.
                                last_path_cells = None
                                print("No path to selected goal yet; retrying...")
                            last_start = cam_row_col
                            last_goal = goal_cell
                            last_path_plan_time = now

                    # Draw path if available.
                    draw_path = path_cells if path_cells else last_path_cells
                    if draw_path:
                        pts = np.array([[c, r] for r, c in draw_path], dtype=np.int32)
                        if pts.shape[0] >= 2:
                            cv2.polylines(map_vis, [pts], False, (255, 255, 0), 1)
                    # Draw goal marker.
                    if goal_cell is not None:
                        gr, gc = goal_cell
                        if 0 <= gr < occ_map.grid_h and 0 <= gc < occ_map.grid_w:
                            cv2.circle(map_vis, (gc, gr), 2, (0, 255, 255), -1)

                    if args.heatmap:
                        heatmap_vis = heatmap_utils.render_heatmap(
                            occ_map,
                            mode=args.heatmap_mode,
                            min_evidence=args.heatmap_min_evidence,
                        )
                        if args.heatmap_window:
                            # Keep normal map colors untouched when showing separate heatmap window.
                            pass
                        else:
                            map_vis = heatmap_utils.blend_with_map(
                                map_vis,
                                heatmap_vis,
                                alpha=args.heatmap_alpha,
                            )
                            cv2.putText(
                                map_vis,
                                f"HEAT:{args.heatmap_mode}",
                                (8, 16),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )

                    # Drive output to RoboRIO (optional).
                    if sd is not None:
                        now = time.time()
                        # Clear ready pulse to create edges like working test scripts.
                        if nt_ready_high and now >= nt_ready_clear_time:
                            sd.putBoolean("Jetson/CommandReady", False)
                            nt_ready_high = False
                        if args.nt_health_debug and (now - nt_last_health_log) >= max(0.2, args.nt_health_period_sec):
                            nt_last_health_log = now
                            nt_health_seq += 1
                            connected = NetworkTables.isConnected()
                            nt_connected_cached = bool(connected)
                            sd.putNumber("Jetson/NTClientSeq", float(nt_health_seq))
                            sd.putNumber("Jetson/NTClientUnix", float(now))
                            sd.putString("Jetson/NTClientName", "zed_ground_wall.py")
                            drive_forward_in = sd.getNumber("Jetson/DriveForward", float("nan"))
                            drive_turn_in = sd.getNumber("Jetson/DriveTurn", float("nan"))
                            speed_in = sd.getNumber("Jetson/Speed", float("nan"))
                            turn_speed_in = sd.getNumber("Jetson/TurnSpeed", float("nan"))
                            ack_seq = sd.getNumber("Jetson/NTServerAckSeq", -1.0)
                            print(
                                f"NT health connected={connected} target={args.roborio_ip} "
                                f"peers=[{nt_connections_summary()}] tx_seq={nt_health_seq} ack_seq={ack_seq:.0f} "
                                f"rx_fwd={drive_forward_in:+.2f} rx_turn={drive_turn_in:+.2f} "
                                f"rx_speed={speed_in:+.2f} rx_turn_speed={turn_speed_in:+.2f}"
                            )
                        elif (now - nt_last_conn_log) >= 2.0:
                            # Periodic lightweight connection status when health debug is off.
                            connected = NetworkTables.isConnected()
                            nt_connected_cached = bool(connected)
                            print(f"NT connected={connected} target={args.roborio_ip}")
                            nt_last_conn_log = now
                        if (now - last_drive_send) >= (1.0 / max(1.0, args.drive_rate_hz)):
                            last_drive_send = now
                            status_target_cell = None
                            status_target_world = None
                            if emergency_stop:
                                send_nt_command(False, 0.0, 0.0, 0.1)
                            elif manual_mode:
                                send_nt_command(
                                    True,
                                    manual_fwd,
                                    manual_turn,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                            elif cam_row_col is None:
                                send_nt_command(False, 0.0, 0.0, 0.1)
                            else:
                                # Pick a waypoint a few steps ahead.
                                if draw_path is not None and len(draw_path) > 0:
                                    wp_index = min(5, len(draw_path) - 1)
                                    wp_rc = draw_path[wp_index]
                                    wp_world = occ_map.grid_to_world(wp_rc[0], wp_rc[1])
                                    if wp_world is None:
                                        continue
                                    tx, tz = wp_world
                                    status_target_cell = wp_rc
                                    status_target_world = (float(tx), float(tz))
                                elif goal_cell is not None:
                                    # Fallback: drive directly toward clicked goal if path is not ready yet.
                                    goal_world_fallback = occ_map.grid_to_world(goal_cell[0], goal_cell[1])
                                    if goal_world_fallback is None:
                                        send_nt_command(False, 0.0, 0.0, 0.1)
                                        continue
                                    tx, tz = goal_world_fallback
                                    status_target_cell = goal_cell
                                    status_target_world = (float(tx), float(tz))
                                else:
                                    send_nt_command(False, 0.0, 0.0, 0.1)
                                    continue
                                # Current pose in world.
                                cx, cz = float(t_world_cam[0]), float(t_world_cam[2])
                                dx = tx - cx
                                dz = tz - cz

                                # Stop if close enough to goal.
                                goal_world = occ_map.grid_to_world(goal_cell[0], goal_cell[1])
                                if goal_world is not None:
                                    gx, gz = goal_world
                                    if math.hypot(gx - cx, gz - cz) <= args.drive_goal_tol_m:
                                        send_nt_command(False, 0.0, 0.0, 0.1)
                                        continue

                                # Heading error from camera forward axis.
                                forward = R_world_cam[:, 2]
                                heading = math.atan2(float(forward[2]), float(forward[0]))
                                if args.drive_heading_flip:
                                    heading += math.pi
                                target = math.atan2(dz, dx)
                                err = target - heading
                                # Wrap to [-pi, pi].
                                while err > math.pi:
                                    err -= 2 * math.pi
                                while err < -math.pi:
                                    err += 2 * math.pi

                                tol = math.radians(max(0.0, args.drive_heading_tol_deg))
                                if abs(err) <= tol:
                                    turn = 0.0
                                else:
                                    turn = max(-1.0, min(1.0, args.drive_turn_k * err))
                                fwd = max(0.0, min(1.0, args.drive_speed))

                                send_nt_command(
                                    True,
                                    fwd,
                                    turn,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                    # Periodically save persistent map to disk.
                    if args.map_save_every > 0 and (time.time() - last_save) >= args.map_save_every:
                        occ_map.save(args.map_save_path)
                        last_save = time.time()
                    # Periodically update and save spatial map (mesh) if enabled.
                    if spatial_enabled and args.spatial_save_path and args.spatial_save_every > 0:
                        if (time.time() - last_spatial_save) >= args.spatial_save_every:
                            ok = zed_utils.update_spatial_map(
                                zed, sl, spatial_mesh, args.spatial_save_path, mesh_filter=args.spatial_filter
                            )
                            if ok:
                                last_spatial_save = time.time()
                                if mesh_viewer is not None:
                                    mesh_viewer.update_from_path(args.spatial_save_path)
                    if mesh_viewer is not None:
                        mesh_viewer.poll()
                else:
                    map_vis = np.zeros((occ_map.grid_h, occ_map.grid_w, 3), dtype=np.uint8)
                    if args.heatmap:
                        heatmap_vis = heatmap_utils.render_heatmap(
                            occ_map,
                            mode=args.heatmap_mode,
                            min_evidence=args.heatmap_min_evidence,
                        )
                        if not args.heatmap_window:
                            map_vis = heatmap_utils.blend_with_map(
                                map_vis,
                                heatmap_vis,
                                alpha=args.heatmap_alpha,
                            )

            # Live visualization (optional)
            if HAS_CV2:
                img = image_left.get_data()
                if img is not None:
                    # Normalize to BGR (3 channels) across SDK image formats.
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif img.ndim == 3:
                        if img.shape[2] == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        elif img.shape[2] == 1:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        elif img.shape[2] > 3:
                            img = img[:, :, :3]
                    else:
                        img = None
                if img is not None:
                    # Build a ground/obstacle mask at the same stride
                    xyz_full = cloud[::stride, ::stride, :3]
                    valid = np.isfinite(xyz_full).all(axis=2)
                    dist_num = (a * xyz_full[:, :, 0] + b * xyz_full[:, :, 1] + c * xyz_full[:, :, 2] + d)
                    denom = np.sqrt(a * a + b * b + c * c)
                    dist_full = np.full_like(dist_num, np.nan, dtype=np.float32)
                    if np.any(valid):
                        dist_full[valid] = (dist_num[valid] / denom).astype(np.float32)

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
                        if not map_window_ready:
                            cv2.setMouseCallback("ZED Occupancy Map (XZ)", on_map_click)
                            map_window_ready = True
                    if args.heatmap and args.heatmap_window and heatmap_vis is not None:
                        heatmap_show = heatmap_vis
                        if args.map_scale > 1:
                            heatmap_show = cv2.resize(
                                heatmap_show,
                                (occ_map.grid_w * args.map_scale, occ_map.grid_h * args.map_scale),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        cv2.imshow("ZED Heatmap (XZ)", heatmap_show)
                cv2.imshow("ZED Drive Status", render_status_panel(cam_row_col))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
                    manual_mode = not manual_mode
                    if manual_mode:
                        # Entering manual mode pauses auto navigation but keeps the last goal.
                        emergency_stop = False
                        manual_fwd = 0.0
                        manual_turn = 0.0
                        print("Manual drive mode: ON (auto paused)")
                    else:
                        print("Manual drive mode: OFF (auto resumed)")
                if key == ord(" "):
                    emergency_stop = True
                    manual_fwd = 0.0
                    manual_turn = 0.0
                now = time.time()
                if key == ord("w"):
                    manual_fwd = max(0.0, min(1.0, args.drive_speed))
                    last_w_time = now
                if key == ord("s"):
                    manual_fwd = -max(0.0, min(1.0, args.drive_speed))
                    last_s_time = now
                if key == ord("a"):
                    manual_turn = -max(0.0, min(1.0, args.drive_speed))
                    last_a_time = now
                if key == ord("d"):
                    manual_turn = max(0.0, min(1.0, args.drive_speed))
                    last_d_time = now
                if key == ord("x"):
                    manual_fwd = 0.0
                    manual_turn = 0.0
                # Hold-to-move: decay to 0 if key not pressed recently.
                if manual_mode:
                    if now - last_w_time > key_hold_timeout and now - last_s_time > key_hold_timeout:
                        manual_fwd = 0.0
                    if now - last_a_time > key_hold_timeout and now - last_d_time > key_hold_timeout:
                        manual_turn = 0.0
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
