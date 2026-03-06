from fastapi import FastAPI
from app.core.settings import settings
from app.routes import router as api_router

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(api_router)