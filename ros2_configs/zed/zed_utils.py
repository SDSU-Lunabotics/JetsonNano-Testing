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
