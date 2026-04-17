from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MapStreamStatus(BaseModel):
    available: bool = False
    source: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    last_frame_ms: Optional[int] = None
    source_timestamp_ms: Optional[int] = None
    frame_seq: int = 0


class MapFrameIngestResponse(BaseModel):
    ok: bool
    timestamp_ms: int
    frame_seq: int


class MapWaypointClickRequest(BaseModel):
    display_x: int = Field(ge=0)
    display_y: int = Field(ge=0)
    source: Optional[str] = None


class MapWaypointCommandResponse(BaseModel):
    ok: bool
    timestamp_ms: int
    command_seq: int


class MapUiControl(BaseModel):
    id: str
    label: str
    command: str
    active: bool = False
    enabled: bool = True


class MapUiStateResponse(BaseModel):
    available: bool = False
    source: Optional[str] = None
    timestamp_ms: Optional[int] = None
    mining_state: Optional[str] = None
    selected_tool: Optional[str] = None
    brush_radius: Optional[int] = None
    controls: list[MapUiControl] = Field(default_factory=list)


class MapUiCommandRequest(BaseModel):
    command: str = Field(
        description=(
            "UI map action. Supported values: paint_obstacle, paint_safe, erase_safe, "
            "clear_all, draw_excav_zone, draw_deposit_zone."
        )
    )
    source: Optional[str] = None


class MapUiCommandResponse(BaseModel):
    ok: bool
    timestamp_ms: int
    command_seq: int
    command: str
