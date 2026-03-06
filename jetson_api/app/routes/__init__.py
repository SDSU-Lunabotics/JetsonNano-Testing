from fastapi import APIRouter

from .health import router as health_router
from .control import router as control_router
from .actuators import router as actuators_router
from .admin import router as admin_router
from .status import router as status_router

router = APIRouter()
router.include_router(health_router)
router.include_router(control_router)
router.include_router(actuators_router)
router.include_router(admin_router)
router.include_router(status_router)