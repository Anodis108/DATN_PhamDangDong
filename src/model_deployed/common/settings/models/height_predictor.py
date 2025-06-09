from __future__ import annotations

from common.bases import BaseModel


class HeightPredictorSettings(BaseModel):
    model_path_linear: str
    model_path_random_forest: str
    model_path_linear_torch: str
    model_path_height_net: str
    mode: str
