from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import requests  # type: ignore
from common.bases import BaseModel
from common.bases import BaseService
from common.settings import Settings

# from io import BytesIO
# import cv2


class PoseDetectorInput(BaseModel):
    img_origin: np.ndarray


class PoseDetectorOutput(BaseModel):
    pose_landmarks: list[list[dict]]
    img_width: float
    img_height: float


class PoseDetector(BaseService):
    settings: Settings

    def process(self, inputs: PoseDetectorInput) -> PoseDetectorOutput:
        _, buffer = cv2.imencode('.jpg', inputs.img_origin)
        file_bytes = BytesIO(buffer.tobytes())
        file_bytes.name = 'image.jpg'
        files = {'file': (file_bytes.name, file_bytes, 'image/jpeg')}
        response = requests.post(
            str(self.settings.host_pose_detector), files=files,
        )

        return PoseDetectorOutput(
            pose_landmarks=response.json()['info']['pose_landmarks'],
            img_width=response.json()['info']['img_width'],
            img_height=response.json()['info']['img_height'],
        )
