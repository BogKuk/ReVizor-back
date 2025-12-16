from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, field_validator

GameType = Literal["low-poly", "indie", "aa", "aaa", "cinematic"]
UsageArea = Literal["background", "prop", "hero"]

class AnalyzeParams(BaseModel):
    game_type: GameType
    usage_area: UsageArea
    extra_params: Optional[Dict[str, Any]] = None

    @field_validator("game_type", mode="before")
    @classmethod
    def _norm_game_type(cls, v):
        return str(v).lower()

    @field_validator("usage_area", mode="before")
    @classmethod
    def _norm_usage_area(cls, v):
        return str(v).lower()
