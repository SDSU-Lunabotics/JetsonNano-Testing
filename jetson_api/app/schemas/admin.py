from __future__ import annotations

from pydantic import BaseModel


class RestartServiceRequest(BaseModel):
    service_name: str


class AdminActionResponse(BaseModel):
    ok: bool
    message: str | None = None