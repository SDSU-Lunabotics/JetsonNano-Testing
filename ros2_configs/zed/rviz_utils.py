import os
import shutil
import subprocess


def launch_rviz(enabled, rviz_config_path):
    if not enabled:
        return
    if shutil.which("rviz2") is None:
        print("rviz2 not found in PATH. Did you source ROS2?")
        return
    if rviz_config_path and os.path.exists(rviz_config_path):
        print(f"Launching rviz2 with config: {rviz_config_path}")
        subprocess.Popen(["rviz2", "-d", rviz_config_path])
        return
    if rviz_config_path:
        print(f"RViz config not found: {rviz_config_path}. Launching default RViz.")
    subprocess.Popen(["rviz2"])
