"""Face detector API format input """
from __future__ import annotations

from typing import List

from common.bases import BaseModel


class BaseResults(BaseModel):
    heights: List[float]
    out_path: str
