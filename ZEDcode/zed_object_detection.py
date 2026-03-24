import cv2
import numpy as np
import pyzed.sl as sl

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720
init_params.coordinate_units = sl.UNIT.UNIT_METER
init_params.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE

err = zed.open(init_params) # err will show if anything goes wrong during cam setup

obj_param = sl.ObjectDetectionParameters()
obj_param.enable_tracking = True
obj_param.detection_model = sl.

