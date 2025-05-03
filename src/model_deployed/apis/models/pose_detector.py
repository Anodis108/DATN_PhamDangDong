"""Face detector API format input """
from __future__ import annotations

from typing import List

from common.bases import BaseModel
from mediapipe.tasks.python.vision.pose_landmarker import Landmark


class APIInput(BaseModel):
    image: List[List[List[int]]]


class APIOutput(BaseModel):
    pose_landmarks: List[List[Landmark]]
    img_width: float
    img_height: float
