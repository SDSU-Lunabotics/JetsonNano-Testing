from fastapi import APIRouter, HTTPException

from app.schemas.network import (
    NetworkVerifyRequest,
    NetworkVerifyResponse,
    NetworkStatusResponse,
)
from app.services.network_service import network_service

router = APIRouter(prefix="/network", tags=["network"])


@router.get("/status", response_model=NetworkStatusResponse)
def get_network_status() -> NetworkStatusResponse:
    return network_service.get_network_status()


@router.post("/verify", response_model=NetworkVerifyResponse)
def verify_network(req: NetworkVerifyRequest) -> NetworkVerifyResponse:
    try:
        return network_service.verify(req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))