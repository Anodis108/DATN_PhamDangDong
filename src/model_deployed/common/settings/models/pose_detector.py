from __future__ import annotations

from common.bases import BaseModel


class PoseDetectorSettings(BaseModel):
    model_path: str
    output_segmentation_masks: bool = False
    num_poses: int = 1
