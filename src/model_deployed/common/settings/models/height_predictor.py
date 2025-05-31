from __future__ import annotations

from common.bases import BaseModel


class HeightPredictorSettings(BaseModel):
    model_path_linear: str
    model_path_random_forest: str
    mode: str
