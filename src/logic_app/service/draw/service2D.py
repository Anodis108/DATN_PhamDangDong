from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from common.bases import BaseModel
from common.bases import BaseService
from common.logs import get_logger
from common.settings import Settings
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

sys.path.append(str(Path(__file__).parent.parent))
logger = get_logger(__name__)


class VisualizationInput(BaseModel):
    image: np.ndarray
    name_image: str
    pose_landmarks_list: list[list[dict]]
    height_cm: list[float]
    bboxes: np.ndarray
    confidences: np.ndarray


class VisualizationOutput(BaseModel):
    annotated_image: np.ndarray
    output_path: Optional[str] = None


class VisualizationService(BaseService):
    settings: Settings

    def _draw_landmarks(self, image: np.ndarray, pose_landmarks_list: list[list[dict]]) -> np.ndarray:
        annotated_image = np.copy(image)
        for pose_landmarks in pose_landmarks_list:
            proto = landmark_pb2.NormalizedLandmarkList()
            proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=lm['x'], y=lm['y'], z=lm['z'],
                )
                for lm in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image, proto, solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        return annotated_image

    def process(self, inputs: VisualizationInput) -> VisualizationOutput:
        try:
            # Vẽ landmarks
            annotated_image = self._draw_landmarks(
                inputs.image, inputs.pose_landmarks_list,
            )

            # Vẽ bounding boxes nếu có YOLO
            if inputs.bboxes is not None and inputs.confidences is not None:
                for box, conf in zip(inputs.bboxes, inputs.confidences):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(
                        annotated_image, (x1, y1),
                        (x2, y2), (0, 255, 0), 2,
                    )
                    cv2.putText(
                        annotated_image, f'box {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                    )

            # Ghi thông tin chiều cao
            text = f'Estimated Height: {inputs.height_cm[0]:.2f} cm'
            cv2.putText(
                annotated_image, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
            )

            # Lưu ảnh
            output_dir = self.settings.draw.output_dir
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/{inputs.name_image}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            cv2.imwrite(output_path, annotated_image)

            return VisualizationOutput(
                annotated_image=annotated_image,
                output_path=output_path,
            )

        except Exception as e:
            logger.error(f'Visualization failed: {str(e)}')
            raise
