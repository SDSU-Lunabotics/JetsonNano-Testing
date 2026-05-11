import json
import time
def write_lidar_pose_json(
    x,
    y,
    yaw_rad,
    path="/tmp/lidar_pose.json",
    *,
    map_x=None,
    map_y=None,
    map_origin_x=None,
    map_origin_y=None,
):
    """
    Write the current pose for lidar integration.

    `x`/`y` remain the absolute ZED world-frame position for compatibility.
    When available, `map_x`/`map_y` expose the same pose in the map-local
    frame that ZEDAuto uses for occupancy rendering.
    """
    payload = {
        "x": float(x),
        "y": float(y),
        "yaw_rad": float(yaw_rad),
        "timestamp": time.time(),
    }
    if map_x is not None:
        payload["map_x"] = float(map_x)
    if map_y is not None:
        payload["map_y"] = float(map_y)
    if map_origin_x is not None:
        payload["map_origin_x"] = float(map_origin_x)
    if map_origin_y is not None:
        payload["map_origin_y"] = float(map_origin_y)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as exc:
        print(f"[LidarPose] Failed to write pose to {path}: {exc}")
import os
import time
import numpy as np


def _quat_to_rotmat(quat):
    q = np.array(quat, dtype=np.float32).reshape(-1)
    if q.size < 4:
        return None
    x, y, z, w = [float(v) for v in q[:4]]
    n = x * x + y * y + z * z + w * w
    if n <= 1e-8:
        return None
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def _orientation_to_rotmat(orientation):
    if orientation is None:
        return None
    try:
        if hasattr(orientation, "to_rotation_matrix"):
            rot = orientation.to_rotation_matrix()
            if hasattr(rot, "r"):
                return np.array(rot.r, dtype=np.float32).reshape(3, 3)
            if hasattr(rot, "get"):
                return np.array(rot.get(), dtype=np.float32).reshape(3, 3)
            return np.array(rot, dtype=np.float32).reshape(3, 3)
    except Exception:
        pass
    try:
        if hasattr(orientation, "get"):
            rot = _quat_to_rotmat(orientation.get())
            if rot is not None:
                return rot
    except Exception:
        pass
    try:
        rot = _quat_to_rotmat(orientation)
        if rot is not None:
            return rot
    except Exception:
        pass
    return None


def get_imu_rotation_with_status(zed, sl, sensors_data, sensor_warned, time_reference=None):
    if sensors_data is None:
        return None, sensor_warned, False

    imu_rot = None
    imu_ok = False
    time_ref = time_reference
    if time_ref is None:
        time_ref = getattr(getattr(sl, "TIME_REFERENCE", None), "IMAGE", None)
    if time_ref is None:
        time_ref = getattr(getattr(sl, "TIME_REFERENCE", None), "CURRENT", None)

    try:
        err = zed.get_sensors_data(sensors_data, time_ref)
        if err != sl.ERROR_CODE.SUCCESS:
            if not sensor_warned:
                print(f"IMU read failed: {err}")
                sensor_warned = True
            return None, sensor_warned, False

        imu_data = None
        if hasattr(sensors_data, "get_imu_data"):
            imu_data = sensors_data.get_imu_data()
        elif hasattr(sensors_data, "imu"):
            imu_data = sensors_data.imu
        if imu_data is None:
            return None, sensor_warned, False

        pose = None
        if hasattr(imu_data, "get_pose"):
            pose = imu_data.get_pose()
        elif hasattr(imu_data, "pose"):
            pose = imu_data.pose

        if pose is not None and hasattr(pose, "get_rotation_matrix"):
            rot = pose.get_rotation_matrix()
            if hasattr(rot, "r"):
                imu_rot = np.array(rot.r, dtype=np.float32).reshape(3, 3)
            elif hasattr(rot, "get"):
                imu_rot = np.array(rot.get(), dtype=np.float32).reshape(3, 3)
            else:
                imu_rot = np.array(rot, dtype=np.float32).reshape(3, 3)

        if imu_rot is None and pose is not None and hasattr(pose, "get_orientation"):
            imu_rot = _orientation_to_rotmat(pose.get_orientation())

        if imu_rot is None and hasattr(imu_data, "get_pose_covariance"):
            # Newer APIs still expose fused orientation through pose; if not present,
            # do not guess from raw gyro/accel.
            imu_rot = None

        if imu_rot is None:
            return None, sensor_warned, False

        if not np.all(np.isfinite(imu_rot)):
            return None, sensor_warned, False

        imu_ok = True
        sensor_warned = False
    except Exception as exc:
        if not sensor_warned:
            print(f"IMU rotation read failed: {exc}")
            sensor_warned = True

    return imu_rot, sensor_warned, imu_ok


def open_zed_camera(sl):
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    last_err = None
    max_attempts = 4

    for attempt in range(1, max_attempts + 1):
        zed = sl.Camera()
        try:
            # Best-effort cleanup in case the SDK object was left half-open.
            zed.close()
        except Exception:
            pass

        err = zed.open(init)
        if err == sl.ERROR_CODE.SUCCESS:
            if attempt > 1:
                print(f"ZED camera opened after retry {attempt}/{max_attempts}.")
            return zed

        last_err = err
        try:
            zed.close()
        except Exception:
            pass

        if attempt < max_attempts:
            print(f"ZED open attempt {attempt}/{max_attempts} failed: {err}. Retrying...")
            time.sleep(1.0)

    raise RuntimeError(f"Failed to open ZED camera after {max_attempts} attempts: {last_err}")


def enable_tracking(zed, sl, area_memory=False, area_load_path=None):
    pose = sl.Pose()
    area_enabled = bool(area_memory or area_load_path)
    enable_fn = None
    if hasattr(zed, "enable_positional_tracking"):
        enable_fn = zed.enable_positional_tracking
    elif hasattr(zed, "enable_tracking"):
        # Older API naming fallback.
        enable_fn = zed.enable_tracking

    if enable_fn is None:
        print("Failed to enable positional tracking: camera API missing tracking enable function")
        return False, pose

    tracking_params = None
    try:
        tracking_params = sl.PositionalTrackingParameters()
        if hasattr(tracking_params, "enable_area_memory"):
            tracking_params.enable_area_memory = area_enabled
        if area_load_path and os.path.exists(area_load_path):
            if hasattr(tracking_params, "area_file_path"):
                tracking_params.area_file_path = area_load_path
            elif hasattr(tracking_params, "set_area_file_path"):
                tracking_params.set_area_file_path(area_load_path)
    except Exception as exc:
        # Don't fail early: some bindings differ on constructor shape.
        print(f"Tracking params unavailable; retrying with default tracking init: {exc}")

    track_err = None
    if tracking_params is not None:
        try:
            track_err = enable_fn(tracking_params)
        except TypeError:
            track_err = None
        except Exception as exc:
            print(f"Tracking init with parameters failed; retrying default init: {exc}")
            track_err = None

    if track_err is None:
        try:
            track_err = enable_fn()
        except Exception as exc:
            print(f"Failed to enable positional tracking: {exc}")
            return False, pose

    if track_err == sl.ERROR_CODE.SUCCESS:
        if area_load_path and os.path.exists(area_load_path):
            print(f"Positional tracking enabled (area memory load: {area_load_path}).")
        elif area_enabled:
            print("Positional tracking enabled (area memory on).")
        else:
            print("Positional tracking enabled.")
        return True, pose

    # If area-memory was requested, retry once without area-memory because
    # stale/incompatible area files can prevent tracking from enabling.
    if area_enabled:
        print(
            f"Tracking failed with area-memory settings ({track_err}); retrying with area-memory disabled."
        )
        retry_err = None
        retry_params = None
        try:
            retry_params = sl.PositionalTrackingParameters()
            if hasattr(retry_params, "enable_area_memory"):
                retry_params.enable_area_memory = False
            if hasattr(retry_params, "area_file_path"):
                retry_params.area_file_path = ""
            elif hasattr(retry_params, "set_area_file_path"):
                retry_params.set_area_file_path("")
        except Exception:
            retry_params = None

        if retry_params is not None:
            try:
                retry_err = enable_fn(retry_params)
            except TypeError:
                retry_err = None
            except Exception:
                retry_err = None

        if retry_err is None:
            try:
                retry_err = enable_fn()
            except Exception as exc:
                print(f"Tracking retry without area-memory failed: {exc}")
                retry_err = None

        if retry_err == sl.ERROR_CODE.SUCCESS:
            print("Positional tracking enabled (area-memory disabled fallback).")
            return True, pose

    print(f"Failed to enable positional tracking: {track_err}")
    return False, pose


def get_world_transform_with_status(zed, sl, pose, pose_warned):
    R_world_cam = None
    t_world_cam = None
    tracking_ok = False
    try:
        pose_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        if pose_state == sl.POSITIONAL_TRACKING_STATE.OK:
            tracking_ok = True
            if hasattr(pose, "get_rotation_matrix"):
                rot = pose.get_rotation_matrix()
                if hasattr(rot, "r"):
                    R_world_cam = np.array(rot.r, dtype=np.float32).reshape(3, 3)
                elif hasattr(rot, "get"):
                    R_world_cam = np.array(rot.get(), dtype=np.float32).reshape(3, 3)
                else:
                    R_world_cam = np.array(rot, dtype=np.float32).reshape(3, 3)
            elif hasattr(pose, "get_orientation"):
                ori = pose.get_orientation()
                if hasattr(ori, "to_rotation_matrix"):
                    rot = ori.to_rotation_matrix()
                    if hasattr(rot, "get"):
                        R_world_cam = np.array(rot.get(), dtype=np.float32).reshape(3, 3)
                    else:
                        R_world_cam = np.array(rot, dtype=np.float32).reshape(3, 3)
            if hasattr(pose, "get_translation"):
                trans = pose.get_translation()
                if hasattr(trans, "get"):
                    t_world_cam = np.array(trans.get(), dtype=np.float32).reshape(3)
                elif hasattr(trans, "x"):
                    t_world_cam = np.array([trans.x, trans.y, trans.z], dtype=np.float32)
        else:
            if not pose_warned:
                print(f"Tracking not OK yet: {pose_state}")
                pose_warned = True
    except Exception as exc:
        if not pose_warned:
            print(f"Pose read failed; falling back to camera frame: {exc}")
            pose_warned = True

    if R_world_cam is None or t_world_cam is None:
        R_world_cam = np.eye(3, dtype=np.float32)
        t_world_cam = np.zeros(3, dtype=np.float32)
    return R_world_cam, t_world_cam, pose_warned, tracking_ok


def get_world_transform(zed, sl, pose, pose_warned):
    # Backward-compatible helper for older callers.
    R_world_cam, t_world_cam, pose_warned, _tracking_ok = get_world_transform_with_status(
        zed, sl, pose, pose_warned
    )
    return R_world_cam, t_world_cam, pose_warned


def _get_enum(sl, names):
    for name in names:
        if hasattr(sl, name):
            return getattr(sl, name)
    return None


def _enum_value(enum_obj, value, default_name):
    if enum_obj is None:
        return None
    key = (value or "").upper()
    if hasattr(enum_obj, key):
        return getattr(enum_obj, key)
    if hasattr(enum_obj, default_name):
        return getattr(enum_obj, default_name)
    return None


def _map_spatial_resolution(sl, value):
    enum_obj = _get_enum(sl, ["SPATIAL_MAP_RESOLUTION", "SPATIAL_MAPPING_RESOLUTION"])
    return _enum_value(enum_obj, value, "MEDIUM")


def _map_spatial_range(sl, value):
    enum_obj = _get_enum(sl, ["SPATIAL_MAP_RANGE", "SPATIAL_MAPPING_RANGE"])
    return _enum_value(enum_obj, value, "MEDIUM")


def enable_spatial_mapping(zed, sl, resolution="medium", mapping_range="medium"):
    try:
        params = sl.SpatialMappingParameters()
        res = _map_spatial_resolution(sl, resolution)
        rng = _map_spatial_range(sl, mapping_range)
        if hasattr(params, "map_resolution") and res is not None:
            params.map_resolution = res
        if hasattr(params, "resolution") and res is not None:
            params.resolution = res
        if hasattr(params, "map_range") and rng is not None:
            params.map_range = rng
        if hasattr(params, "range") and rng is not None:
            params.range = rng
        err = zed.enable_spatial_mapping(params)
        if err == sl.ERROR_CODE.SUCCESS:
            print("Spatial mapping enabled.")
            return True, sl.Mesh()
        print(f"Failed to enable spatial mapping: {err}")
    except Exception as exc:
        print(f"Failed to enable spatial mapping: {exc}")
    return False, None


def _map_mesh_filter(sl, value):
    if not hasattr(sl, "MESH_FILTER"):
        return None
    v = (value or "").lower()
    if v == "low":
        return sl.MESH_FILTER.LOW
    if v == "high":
        return sl.MESH_FILTER.HIGH
    if v == "medium":
        return sl.MESH_FILTER.MEDIUM
    return None


def update_spatial_map(zed, sl, mesh, save_path, mesh_filter="none"):
    if mesh is None:
        return False
    try:
        err = None
        if hasattr(zed, "request_spatial_map_async"):
            zed.request_spatial_map_async()
            if hasattr(zed, "get_spatial_map_async"):
                err = zed.get_spatial_map_async(mesh)
            elif hasattr(zed, "retrieve_spatial_map_async"):
                err = zed.retrieve_spatial_map_async(mesh)
        elif hasattr(zed, "request_spatial_map"):
            zed.request_spatial_map()
            if hasattr(zed, "retrieve_spatial_map"):
                err = zed.retrieve_spatial_map(mesh)
        elif hasattr(zed, "extract_whole_spatial_map"):
            err = zed.extract_whole_spatial_map(mesh)

        if err is not None and err != sl.ERROR_CODE.SUCCESS:
            return False
        filt = _map_mesh_filter(sl, mesh_filter)
        if filt is not None and hasattr(mesh, "filter") and hasattr(sl, "MeshFilterParameters"):
            mesh.filter(sl.MeshFilterParameters(filt))
        if hasattr(mesh, "save"):
            mesh.save(save_path)
            print(f"Saved spatial mesh: {save_path}")
            return True
    except Exception as exc:
        print(f"Spatial map update failed: {exc}")
    return False


def disable_spatial_mapping(zed):
    try:
        zed.disable_spatial_mapping()
    except Exception:
        pass


def save_area_memory(zed, sl, path):
    if not path:
        return False
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if hasattr(zed, "save_area_map"):
            err = zed.save_area_map(path)
            if err == sl.ERROR_CODE.SUCCESS:
                print(f"Saved area memory: {path}")
                return True
            print(f"save_area_map failed: {err}")
            return False
    except Exception as exc:
        print(f"Failed to save area memory: {exc}")
    return False
