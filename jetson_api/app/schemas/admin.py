from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


class RestartServiceRequest(BaseModel):
    service_name: str


class AdminActionResponse(BaseModel):
    ok: bool
    message: Optional[str] = None