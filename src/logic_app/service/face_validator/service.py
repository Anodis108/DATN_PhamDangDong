from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from common.bases import BaseModel
from common.bases import BaseService
from common.logs import get_logger
from common.settings import Settings
from infrastructure.text_ocr import FaceLandMark
from infrastructure.text_ocr import FaceLandMarkInput

sys.path.append(str(Path(__file__).parent.parent))

logger = get_logger(__name__)


class FaceValidatorInput(BaseModel):
    """Represents input data for face validation."""
    image: np.ndarray
    # Array of bounding boxes (N, 5) -> (x1, y1, x2, y2, score)
    bboxes: np.ndarray
    kpss: Optional[np.ndarray]  # Array of keypoints (N, 5, 2) or None


class FaceValidatorOutput(BaseModel):
    """Represents the output after processing face validation."""
    bboxes: Optional[np.ndarray]  # The largest bounding box or None
    kpss: Optional[np.ndarray]  # Corresponding keypoints or None


class FaceValidatorService(BaseService):
    settings: Settings

    @property
    def _get_face_landmark(self) -> FaceLandMark:
        return FaceLandMark(settings=self.settings)

    def process(self, inputs: FaceValidatorInput) -> FaceValidatorOutput:
        """
        Validates input bounding boxes (bboxes) and keypoints (kpss).
        If no valid faces are found, returns None.
        Otherwise, selects the face with the largest area and highest confidence score,
        and ensures face angle in [-15, 15] degrees.
        """
        if inputs.bboxes is None or inputs.bboxes.size == 0:
            logger.warning('No bounding boxes provided.')
            return FaceValidatorOutput(bboxes=None, kpss=None)

        max_area: float = 0
        best_score: float = 0
        largest_face_idx: int = -1

        for i, bbox in enumerate(inputs.bboxes):
            x1, y1, x2, y2, score = bbox

            # Check if the bbox is valid
            if x2 <= x1 or y2 <= y1 or score < self.settings.embedding_thresh:
                continue

            area: float = (x2 - x1) * (y2 - y1)

            # Ensure face angle in [-15, 15] degrees
            try:
                face_landmark_output = self._get_face_landmark.process(
                    inputs=FaceLandMarkInput(
                        image=inputs.image,
                        bbox=[x1, y1, x2, y2],
                    ),
                )
                face_angle = face_landmark_output.pred['face_angle'][-1]
                if not (-15 <= face_angle <= 15):
                    continue
            except Exception as e:
                logger.error(f'Failed to process face FaceLandMark model: {e}')

            # Select the largest bbox, or if equal, choose the one with the highest score
            if area > max_area or (area == max_area and score > best_score):
                max_area = area
                best_score = score
                largest_face_idx = i

        if largest_face_idx == -1:
            return FaceValidatorOutput(bboxes=None, kpss=None)

        largest_bbox: np.ndarray = inputs.bboxes[largest_face_idx]
        largest_kpss: Optional[np.ndarray] = (
            inputs.kpss[largest_face_idx] if inputs.kpss is not None else None
        )

        face_aligner = FaceLandMark(settings=self.settings)
        angle = face_aligner.process(
            FaceLandMarkInput(
                image=inputs.image,
                bbox=largest_bbox.tolist(),  # type: ignore
            ),
        )
        if angle.pred <= -20 and angle.pred >= 20:
            return FaceValidatorOutput(bboxes=None, kpss=None)

        return FaceValidatorOutput(bboxes=largest_bbox, kpss=largest_kpss)
