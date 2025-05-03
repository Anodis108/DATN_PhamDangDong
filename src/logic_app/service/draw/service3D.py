from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
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
    detection_result: mp.tasks.python.vision.PoseLandmarkerResult
    height_cm: float
    bboxes: Optional[np.ndarray] = None
    confidences: Optional[np.ndarray] = None


class VisualizationOutput(BaseModel):
    annotated_image: np.ndarray
    output_path: Optional[str] = None


class VisualizationService(BaseService):
    settings: Settings

    def __init__(self, settings: Settings):
        self.settings = settings

    def _draw_landmarks(self, image: np.ndarray, detection_result: mp.tasks.python.vision.PoseLandmarkerResult) -> np.ndarray:
        annotated_image = np.copy(image)
        for pose_landmarks in detection_result.pose_landmarks:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image, pose_landmarks_proto, solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        return annotated_image

    def process(self, inputs: VisualizationInput) -> VisualizationOutput:
        try:
            annotated_image = self._draw_landmarks(
                inputs.image, inputs.detection_result,
            )

            # Draw YOLO bounding boxes (if provided)
            if inputs.bboxes is not None and inputs.confidences is not None:
                for box, conf in zip(inputs.bboxes, inputs.confidences):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(
                        annotated_image, (x1, y1),
                        (x2, y2), (0, 255, 0), 2,
                    )
                    cv2.putText(
                        annotated_image, f'Black box {conf:.2f}', (
                            x1, y1 - 10,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                    )

            # Draw height text
            text = f'Estimated Height: {inputs.height_cm:.2f} cm'
            cv2.putText(
                annotated_image, text, (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

            output_path = None
            if inputs.bboxes is None:  # Image mode
                from datetime import datetime
                output_dir = self.settings.get(
                    'output_dir', '3D_pose3_pose4_data/pose4_rotate/output4_rotate',
                )
                output_path = f"{output_dir}/final_image_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                cv2.imwrite(output_path, annotated_image)

                # Resize and display
                screen_res = (1280, 720)
                scale = min(
                    screen_res[0] / annotated_image.shape[1],
                    screen_res[1] / annotated_image.shape[0],
                )
                resized_image = cv2.resize(
                    annotated_image, (
                        int(annotated_image.shape[1] * scale),
                        int(annotated_image.shape[0] * scale),
                    ),
                )
                cv2.putText(
                    resized_image, text, (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                )
                cv2.imshow('Final Result', resized_image)
                cv2.waitKey(100000)
                cv2.destroyAllWindows()
            else:  # Camera mode
                cv2.imshow('Camera Output', annotated_image)

            return VisualizationOutput(annotated_image=annotated_image, output_path=output_path)

        except Exception as e:
            logger.error(f'Visualization failed: {str(e)}')
            raise
