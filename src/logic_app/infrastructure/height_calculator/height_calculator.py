from __future__ import annotations

from typing import List

import requests  # type: ignore
from common.bases import BaseModel
from common.bases import BaseService
from common.logs import get_logger
from common.settings import Settings

# from typing import Any
logger = get_logger(__name__)


class HeightCalInput(BaseModel):
    landmarks: List[List[dict]]
    img_width: float
    img_height: float
    px_per_cm: float


class HeightCalOutput(BaseModel):
    heights: List[float]
    distances: List[List[float]]
    cm_direct: List[float]
    cm_sum: List[float]
    diffs: List[float]


class HeightCal(BaseService):
    settings: Settings

    def process(self, inputs: HeightCalInput) -> HeightCalOutput:
        payload = {
            'landmarks': inputs.landmarks,
            'img_width': inputs.img_width,
            'img_height': inputs.img_height,
            'px_per_cm': inputs.px_per_cm,
        }
        response = requests.post(
            str(self.settings.host_height_calculator), json=payload,
        )

        return HeightCalOutput(
            heights=response.json()['info']['heights'],
            distances=response.json()['info']['distances'],
            cm_direct=response.json()['info']['cm_direct'],
            cm_sum=response.json()['info']['cm_sum'],
            diffs=response.json()['info']['diffs'],
        )
