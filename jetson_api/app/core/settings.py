from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    app_name: str = "Jetson Lunabotics API"
    app_version: str = "1.0.0"

    host: str = "0.0.0.0"
    port: int = 8000

    jetson_ip: str = "10.0.8.100"
    roborio_ip: str = "10.0.9.2"
    roborio_bridge_url: str = "http://127.0.0.1:8001"
    camera_host: str = "10.0.8.102" # Update with actual IPs
    lidar_host: str = "10.0.8.103" # Update with actual IPs
    lidar_backend: str = "unitree"
    lidar_tcp_host: str = "127.0.0.1"
    lidar_data_port: int = 9876
    lidar_command_port: int = 9877
    lidar_mode: str = "2d"
    lidar_frame_id: str = "unitree_l2"
    lidar_status_ttl_ms: int = 2000
    lidar_monitor_interval_ms: int = 500
    lidar_bridge_command: str = "cd ./lidar && sudo ./lidar_bridge"
    lidar_visualization_command: str = "cd ./lidar && python3 lidar_visualization.py"
    camera_backend: str = "auto"
    camera_device_index: int = 0
    camera_capture_fps: int = 20
    camera_jpeg_quality: int = 80
    camera_status_ttl_ms: int = 2000
    camera_snapshot_cache_ms: int = 200
    camera_stream_interval_ms: int = 120
    camera_worker_retry_ms: int = 1000
    network_status_ttl_ms: int = 5000
    throughput_test_enabled: bool = True
    throughput_test_port: int = 5201
    throughput_test_duration_s: int = 1
    roborio_estimate_ping_count: int = 5
    roborio_estimate_timeout_ms: int = 2000
    
    max_latency_ms: float = 200.0 # Update with acceptable latency threshold
    max_packet_loss_pct: float = 5.0 # Update with acceptable packet loss threshold

    dry_run: bool = True
    allow_reboot: bool = False
    wireless_state_file: str = str(Path(__file__).resolve().parents[2] / "data" / "wireless_state.json")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
