from __future__ import annotations

from functools import cached_property
from typing import List

import cv2
import mediapipe as mp
import numpy as np
from common.bases import BaseModel
from common.bases import BaseService
from common.logs.logs import get_logger
from common.settings import Settings
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

logger = get_logger(__name__)


class PoseDetectorModelInput(BaseModel):
    img: np.ndarray  # Input hình ảnh dạng numpy array


class PoseDetectorModelOutput(BaseModel):
    # Danh sách các landmarks cho từng người (List người x List điểm)
    pose_landmarks: List[List[dict]]
    img_width: float
    img_height: float


class PoseDetectorModel(BaseService):
    settings: Settings

    @cached_property
    def model_loaded(self):
        # Load pose detection model
        base_options = python.BaseOptions(
            model_asset_path=self.settings.pose_detector.model_path,
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
        )
        return vision.PoseLandmarker.create_from_options(options)

    async def process(self, inputs: PoseDetectorModelInput) -> PoseDetectorModelOutput:
        # Gọi forward để trích xuất pose landmarks
        pose_landmarks = self.forward(inputs.img)
        serialized_landmarks = [
            [
                {'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in landmarks
            ] for landmarks in pose_landmarks
        ]

        img_h, img_w = inputs.img.shape[:2]
        return PoseDetectorModelOutput(
            pose_landmarks=serialized_landmarks,
            img_height=float(img_h),
            img_width=float(img_w),
        )

    def forward(self, img: np.ndarray) -> List[List[NormalizedLandmark]]:
        """
        Thực hiện phát hiện pose landmark và trả về danh sách landmark cho từng người.

        Args:
            img (np.ndarray): Ảnh đầu vào (BGR).

        Returns:
            List[List[Landmark]]: Danh sách landmark của từng người.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result: PoseLandmarkerResult = self.model_loaded.detect(
            image,
        )

        if not detection_result.pose_landmarks:
            logger.warning('No pose landmarks detected.')
            return []

        return detection_result.pose_landmarks
