"""Face detector API format input """
from __future__ import annotations

from typing import List

from common.bases import BaseModel


class APIOutput(BaseModel):
    pose_landmarks: List[List[dict]]
    img_width: float
    img_height: float
