"""Face detector API format input """
from __future__ import annotations

from typing import Any
from typing import List

from common.bases import BaseModel


class APIOutput(BaseModel):
    bboxes: List[List[Any]]
    scores: List[float]
