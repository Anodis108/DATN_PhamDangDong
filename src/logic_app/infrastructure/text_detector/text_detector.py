from __future__ import annotations

import numpy as np
import requests  # type: ignore
from common.bases import BaseModel
from common.bases import BaseService
from common.settings import Settings

# from io import BytesIO
# import cv2


class TextDetectorInput(BaseModel):
    img_origin: np.ndarray


class TextDectorOutput(BaseModel):
    class_list: list
    bboxes_list: list[list]
    conf_list: list


class TextDetector(BaseService):
    settings: Settings

    def process(self, inputs: TextDetectorInput) -> TextDectorOutput:
        payload = {
            'image': inputs.img_origin.tolist(),
        }
        response = requests.post(
            str(self.settings.host_text_detector), json=payload,
        )

        return TextDectorOutput(
            class_list=response.json()['info']['classes'],
            bboxes_list=response.json()['info']['bboxes'],
            conf_list=response.json()['info']['confs'],
        )
