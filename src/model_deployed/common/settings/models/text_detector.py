from __future__ import annotations

from common.bases import BaseModel


class TextDetectorSettings(BaseModel):
    model_path: str 
    conf: float
    # nms_thresh: float
    # input_mean: float
    # input_std: float
    # anchor_ratio: float
    # num_anchors: int
