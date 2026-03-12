from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Jetson Lunabotics API"
    app_version: str = "1.0.0"

    host: str = "0.0.0.0"
    port: int = 8000

    jetson_ip: str = "10.0.8.101"
    roborio_ip: str = "10.0.9.2"
    camera_host: str = "10.0.8.102" # Update with actual IPs
    lidar_host: str = "10.0.8.103" # Update with actual IPs
    
    max_latency_ms: float = 200.0 # Update with acceptable latency threshold
    max_packet_loss_pct: float = 5.0 # Update with acceptable packet loss threshold

    dry_run: bool = True
    allow_reboot: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()