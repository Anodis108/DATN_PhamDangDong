"""Face detector API format input """
from __future__ import annotations

from typing import List

from common.bases import BaseModel


class APIInput(BaseModel):
    image: List[List[List[int]]]


class APIOutput(BaseModel):
    classes: List[str]
    bboxes: List[List[float]]
    confs: List[float]
