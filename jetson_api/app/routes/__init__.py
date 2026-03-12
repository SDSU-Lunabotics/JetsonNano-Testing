from fastapi import APIRouter

from .health import router as health_router
from .control import router as control_router
from .actuators import router as actuators_router
from .admin import router as admin_router
from .status import router as status_router
from .network import router as network_router
from .battery import router as battery_router
from .motors import router as motors_router
from .camera import router as camera_router
from .lidar import router as lidar_router
from .scripts import router as scripts_router
from .admin import router as admin_router

router = APIRouter()

router.include_router(health_router)
router.include_router(control_router)
router.include_router(actuators_router)
router.include_router(admin_router)
router.include_router(status_router)
router.include_router(network_router)
router.include_router(battery_router)
router.include_router(motors_router)
router.include_router(camera_router)
router.include_router(lidar_router)
router.include_router(scripts_router)
router.include_router(admin_router)