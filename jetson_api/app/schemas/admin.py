from typing import Optional
from pydantic import BaseModel


class RestartServiceRequest(BaseModel):
    service_name: str


class AdminActionResponse(BaseModel):
    ok: bool
    message: str
    timestamp_ms: int
    detail: Optional[str] = None