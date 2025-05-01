from __future__ import annotations

from common.bases import BaseModel


class BoxDetectorSettings(BaseModel):
    model_path: str
    conf: float
    base_h: float = 30.5 
