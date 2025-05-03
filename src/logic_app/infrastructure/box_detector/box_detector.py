from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import requests  # type: ignore
from common.bases import BaseModel
from common.bases import BaseService
from common.settings import Settings


class BoxDetectorInput(BaseModel):
    image: np.ndarray


class BoxDectorOutput(BaseModel):
    bboxes: list[list[float]]
    scores: list[float]
    pixel_per_cm: float


class BoxDetector(BaseService):
    settings: Settings

    def process(self, inputs: BoxDetectorInput) -> BoxDectorOutput:
        _, buffer = cv2.imencode('.jpg', inputs.image)
        file_bytes = BytesIO(buffer.tobytes())
        file_bytes.name = 'image.jpg'
        files = {'file': (file_bytes.name, file_bytes, 'image/jpeg')}
        response = requests.post(
            str(self.settings.host_box_detector), files=files,
        )

        return BoxDectorOutput(
            bboxes=response.json()['info']['bboxes'],
            scores=response.json()['info']['scores'],
            pixel_per_cm=response.json()['info']['pixel_per_cm'],
        )
