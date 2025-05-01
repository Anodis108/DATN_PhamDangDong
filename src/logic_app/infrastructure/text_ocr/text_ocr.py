from __future__ import annotations

from typing import Any
from typing import List
from typing import Union

import numpy as np
import requests  # type: ignore
from common.bases import BaseModel
from common.bases import BaseService
from common.logs import get_logger
from common.settings import Settings

# from typing import Any
logger = get_logger(__name__)


class TextOCRInput(BaseModel):
    img: np.ndarray  # Ảnh đầu vào
    class_list: List[str]
    bboxes_list: List[List[Union[float, int]]]


class TextOCROutput(BaseModel):
    results: List[Any]


class TextOCR(BaseService):
    settings: Settings

    def process(self, inputs: TextOCRInput) -> TextOCROutput:
        payload = {
            'img': inputs.img.tolist(),  # Chuyển ảnh NumPy sang list
            'classes': inputs.class_list,  # Danh sách class
            'bboxes': inputs.bboxes_list,  # Danh sách tọa độ vùng văn bản
        }
        response = requests.post(
            str(self.settings.host_text_ocr), json=payload,
        )

        return TextOCROutput(results=response.json()['info']['info_text'])
