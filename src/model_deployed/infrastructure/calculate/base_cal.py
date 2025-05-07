from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import List

from common.bases import BaseModel
from common.bases import BaseService
from common.logs.logs import get_logger
from common.settings import Settings
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

logger = get_logger(__name__)


class CalHeightInput(BaseModel):
    # Danh sách các bộ điểm pose landmarks (ví dụ: nhiều người, hoặc nhiều khung hình)
    landmarks: List[List[NormalizedLandmark]]
    img_width: float                  # Chiều rộng ảnh gốc (pixel)
    img_height: float                 # Chiều cao ảnh gốc (pixel)
    px_per_cm: float                  # Tỷ lệ quy đổi pixel → cm


class CalHeightOutput(BaseModel):
    # Chiều cao (theo pixel) đã tính được cho từng người
    heights: List[float]
    # Danh sách độ dài từng phần cơ thể (7 đoạn) cho mỗi người
    distances: List[List[float]]
    # Chiều cao tính trực tiếp (tổng đoạn) sau khi quy đổi ra cm
    cm_direct: List[float]
    cm_sum: List[float]          # Tổng các đoạn cơ thể sau khi quy đổi ra cm
    # Sai số giữa chiều cao trực tiếp và tổng từng đoạn (cm)
    diffs: List[float]


class CalHeight(ABC, BaseService):
    settings: Settings

    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    async def process(self, inputs: CalHeightInput) -> CalHeightOutput:
        ...

    @classmethod
    def get_service(cls, settings: Settings) -> CalHeight:
        mode = settings.height_calculator.mode.upper()
        if mode == '3D':
            from .height_3d import CalHeight3D
            return CalHeight3D(settings=settings)
        elif mode == '2D':
            from .height_2d import CalHeight2D
            return CalHeight2D(settings=settings)
        else:
            raise ValueError(f'Unknown mode: {mode}')
