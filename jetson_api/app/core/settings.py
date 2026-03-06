from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Jetson Lunabotics API"
    app_version: str = "1.0.0"

    host: str = "0.0.0.0"
    port: int = 8000

    roborio_ip: str = "10.0.9.2"

    dry_run: bool = True
    allow_reboot: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()