from fastapi import APIRouter
from app.services.state_service import now_ms

router = APIRouter(tags=["health"])


# proves the jetson server is alive and reachable
@router.get("/ping")
def ping():
    return {"ok": True, "service": "jetson", "timestamp_ms": now_ms()}