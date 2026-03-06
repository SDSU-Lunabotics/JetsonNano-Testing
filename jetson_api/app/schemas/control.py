from pydantic import BaseModel, Field


class LedRequest(BaseModel):
    state: bool
    override: bool = True


class LedResponse(BaseModel):
    ok: bool
    state: bool
    override: bool
    timestamp_ms: int


class DriveForwardRequest(BaseModel):
    duration: float = Field(default=3.0, gt=0.0, le=30.0)
    speed: float = Field(default=0.6, ge=-1.0, le=1.0)


class DriveForwardResponse(BaseModel):
    ok: bool
    action: str
    duration: float
    speed: float
    timestamp_ms: int


class EstopRequest(BaseModel):
    engage: bool = True


class EstopResponse(BaseModel):
    ok: bool
    estop: bool
    timestamp_ms: int