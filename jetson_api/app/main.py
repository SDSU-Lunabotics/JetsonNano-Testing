from fastapi import FastAPI
from app.core.settings import settings
from app.routes import router as api_router
from app.services.camera_service import camera_service
from app.services.lidar_service import lidar_service

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(api_router)


@app.on_event("startup")
def startup() -> None:
    if settings.camera_autostart:
        camera_service.start()
    lidar_service.start()


@app.on_event("shutdown")
def shutdown() -> None:
    camera_service.stop()
    lidar_service.stop()
