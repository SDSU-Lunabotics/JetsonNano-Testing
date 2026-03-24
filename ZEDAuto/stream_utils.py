import cv2
import numpy as np


class GstUdpStreamer:
    def __init__(self, host, port=5600, fps=15, bitrate_kbps=2500):
        self.host = host
        self.port = int(port)
        self.fps = float(fps)
        self.bitrate_kbps = int(bitrate_kbps)
        self.writer = None
        self.width = None
        self.height = None
        self.pipeline = None

    def _pipelines(self, width, height):
        caps = f"video/x-raw,format=BGR,width={width},height={height},framerate={int(self.fps)}/1"
        sw = (
            "appsrc ! "
            f"{caps} ! "
            "videoconvert ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={self.bitrate_kbps} key-int-max=30 ! "
            "rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={self.host} port={self.port} sync=false async=false"
        )
        hw = (
            "appsrc ! "
            f"{caps} ! "
            "videoconvert ! "
            "video/x-raw,format=I420 ! "
            "nvvideoconvert ! "
            f"nvv4l2h264enc bitrate={self.bitrate_kbps * 1000} iframeinterval=30 insert-sps-pps=true ! "
            "h264parse ! "
            "rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={self.host} port={self.port} sync=false async=false"
        )
        # Try hardware first on Jetson, then software fallback.
        return [hw, sw]

    def open(self, width, height):
        self.close()
        self.width = int(width)
        self.height = int(height)
        for pipeline in self._pipelines(self.width, self.height):
            writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, self.fps, (self.width, self.height), True)
            if writer.isOpened():
                self.writer = writer
                self.pipeline = pipeline
                print(f"GStreamer stream active: udp://{self.host}:{self.port}")
                return True
        print("Failed to open GStreamer UDP stream pipeline.")
        return False

    def write(self, frame):
        if frame is None:
            return
        if frame.ndim != 3:
            return
        h, w = frame.shape[:2]
        if self.writer is None:
            if not self.open(w, h):
                return
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
