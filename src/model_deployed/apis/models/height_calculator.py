"""Face detector API format input """
from __future__ import annotations

from typing import Any
from typing import List

from common.bases import BaseModel


class APIInput(BaseModel):
    landmarks: List[List[Any]]
    img_width: float
    img_height: float
    px_per_cm: float


class APIOutput(BaseModel):
    heights: List[float]
    distances: List[List[float]]
    cm_direct: List[float]
    cm_sum: List[float]
    diffs: List[float]
