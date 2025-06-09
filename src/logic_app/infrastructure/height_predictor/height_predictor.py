from __future__ import annotations

from typing import List

import requests  # type: ignore
from common.bases import BaseModel
from common.bases import BaseService
from common.logs import get_logger
from common.settings import Settings

# from typing import Any
logger = get_logger(__name__)


class HeightPredInput(BaseModel):
    x: List[List[float]]


class HeightPredOutput(BaseModel):
    pred: List[float]


class HeightPred(BaseService):
    settings: Settings

    def process(self, inputs: HeightPredInput) -> HeightPredOutput:
        payload = {
            'x': inputs.x,
        }
        response = requests.post(
            str(self.settings.host_height_predictor), json=payload,
        )

        return HeightPredOutput(
            pred=response.json()['info']['pred'],
        )
