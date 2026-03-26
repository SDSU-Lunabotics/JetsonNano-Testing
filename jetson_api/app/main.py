from fastapi import FastAPI
from app.core.settings import settings
from app.routes import router as api_router
from app.services.lidar_service import lidar_service

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(api_router)


@app.on_event("startup")
def startup() -> None:
    lidar_service.start()


@app.on_event("shutdown")
def shutdown() -> None:
    lidar_service.stop()
