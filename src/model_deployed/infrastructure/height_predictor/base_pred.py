from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import List

from common.bases import BaseModel
from common.bases import BaseService
from common.logs.logs import get_logger
from common.settings import Settings

logger = get_logger(__name__)


class HeightPredictorModelInput(BaseModel):
    x: List[List[float]]  # Danh sách nhiều vector đặc trưng


class HeightPredictorModelOutput(BaseModel):
    pred: List[float]  # Danh sách nhiều kết quả dự đoán


class HeightPredictorModel(BaseService, ABC):
    settings: Settings

    @abstractmethod
    async def process(self, inputs: HeightPredictorModelInput) -> HeightPredictorModelOutput:
        ...

    @classmethod
    def get_service(cls, settings: Settings) -> HeightPredictorModel:
        mode = settings.height_predictor.mode.upper()
        if mode == 'LINEAR':
            from .linear_reg import HeightPredictorModelLinear
            return HeightPredictorModelLinear(settings=settings)
        elif mode == 'RANDOM_FOREST':
            from .random_forest import HeightPredictorModelRandomForest
            return HeightPredictorModelRandomForest(settings=settings)
        else:
            raise ValueError(f'Unknown mode: {mode}')
