#!/usr/bin/env python3
import sys
import time

import numpy as np
import pyzed.sl as sl

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

WIDTH = 1280
HEIGHT = 720
FPS = 20
HOST = "192.168.1.1"
PORT = 5600

def build_pipeline():
    # appsrc expects BGR frames; we encode to H.264 and RTP pay to UDP
    pipeline_str = (
        "appsrc name=src is-live=true block=true format=time do-timestamp=true "
        f"caps=video/x-raw,format=BGR,width={WIDTH},height={HEIGHT},framerate={FPS}/1 "
        "! videoconvert "
        "! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2500 key-int-max=20 "
        "! h264parse "
        "! rtph264pay pt=96 config-interval=1 "
        f"! udpsink host={HOST} port={PORT} sync=false"
    )
    return Gst.parse_launch(pipeline_str)

def main():
    # ---- ZED init ----
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("ZED open failed:", err)
        return 1

    runtime = sl.RuntimeParameters()
    image_left = sl.Mat()

    # ---- GStreamer pipeline ----
    pipeline = build_pipeline()
    appsrc = pipeline.get_by_name("src")
    pipeline.set_state(Gst.State.PLAYING)

    frame_period = 1.0 / FPS
    last = time.time()

    try:
        while True:
            # simple pacing to FPS
            now = time.time()
            dt = now - last
            if dt < frame_period:
                time.sleep(frame_period - dt)
            last = time.time()

            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            frame_rgba = image_left.get_data()  # HxWx4, BGRA
            frame_bgr = frame_rgba[:, :, :3].copy()  # BGR

            # Push into appsrc
            buf = Gst.Buffer.new_allocate(None, frame_bgr.nbytes, None)
            buf.fill(0, frame_bgr.tobytes())
            # timestamps are handled by do-timestamp=true, but keeping duration helps
            buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, FPS)

            ret = appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                print("push-buffer returned", ret)
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            appsrc.emit("end-of-stream")
        except Exception:
            pass
        pipeline.set_state(Gst.State.NULL)
        zed.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())
