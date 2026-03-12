import numpy as np


def open_zed_camera(sl):
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    zed = sl.Camera()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")
    return zed


def enable_tracking(zed, sl):
    pose = sl.Pose()
    try:
        tracking_params = sl.PositionalTrackingParameters()
        track_err = zed.enable_positional_tracking(tracking_params)
        if track_err == sl.ERROR_CODE.SUCCESS:
            print("Positional tracking enabled.")
            return True, pose
        print(f"Failed to enable positional tracking: {track_err}")
    except Exception as exc:
        print(f"Failed to enable positional tracking: {exc}")
    return False, pose


def get_world_transform(zed, sl, pose, pose_warned):
    R_world_cam = None
    t_world_cam = None
    try:
        pose_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        if pose_state == sl.POSITIONAL_TRACKING_STATE.OK:
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


def update_spatial_map(zed, sl, mesh, save_path):
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
        if hasattr(mesh, "filter") and hasattr(sl, "MeshFilterParameters"):
            mesh.filter(sl.MeshFilterParameters(sl.MESH_FILTER.MEDIUM))
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
