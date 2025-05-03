from __future__ import annotations

import numpy as np
import requests  # type: ignore
from common.bases import BaseModel
from common.bases import BaseService
from common.settings import Settings
from mediapipe.tasks.python.vision.pose_landmarker import Landmark

# from io import BytesIO
# import cv2


class PoseDetectorInput(BaseModel):
    img_origin: np.ndarray


class PoseDetectorOutput(BaseModel):
    pose_landmarks: list[list[Landmark]]
    img_width: float
    img_height: float


class PoseDetector(BaseService):
    settings: Settings

    def process(self, inputs: PoseDetectorInput) -> PoseDetectorOutput:
        payload = {
            'image': inputs.img_origin.tolist(),
        }
        response = requests.post(
            str(self.settings.host_pose_detector), json=payload,
        )

        return PoseDetectorOutput(
            pose_landmarks=response.json()['info']['pose_landmarks'],
            img_width=response.json()['info']['img_width'],
            img_height=response.json()['info']['img_height'],
        )
