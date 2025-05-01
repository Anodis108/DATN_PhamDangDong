from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import requests  # type: ignore
from common.bases import BaseModel
from common.bases import BaseService
from common.settings import Settings


class CardDetectorInput(BaseModel):
    image: np.ndarray


class CardDectorOutput(BaseModel):
    bboxes: list[list[float]]
    scores: list[float]


class CardDetector(BaseService):
    settings: Settings

    def process(self, inputs: CardDetectorInput) -> CardDectorOutput:
        _, buffer = cv2.imencode('.jpg', inputs.image)
        file_bytes = BytesIO(buffer.tobytes())
        file_bytes.name = 'image.jpg'
        files = {'file': (file_bytes.name, file_bytes, 'image/jpeg')}
        response = requests.post(
            str(self.settings.host_card_detector), files=files,
        )

        return CardDectorOutput(
            bboxes=response.json()['info']['bboxes'],
            scores=response.json()['info']['scores'],
        )
