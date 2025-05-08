from __future__ import annotations

import numpy as np
from common.bases import BaseModel
from common.bases import BaseService
from common.logs import get_logger
from common.settings import Settings
from infrastructure.box_detector import BoxDetector
from infrastructure.box_detector import BoxDetectorInput
from infrastructure.height_calculator import HeightCal
from infrastructure.height_calculator import HeightCalInput
from infrastructure.height_predictor import HeightPred
from infrastructure.pose_detector import PoseDetector
from infrastructure.pose_detector import PoseDetectorInput

logger = get_logger(__name__)


class HeightInput(BaseModel):
    image: np.ndarray


class HeightOutput(BaseModel):
    # status: bool
    results: list[float]


class HeightService(BaseService):
    settings: Settings

    @property
    def _get_box_detector(self) -> BoxDetector:
        return BoxDetector(settings=self.settings)

    @property
    def _get_pose_detector(self) -> PoseDetector:
        return PoseDetector(settings=self.settings)

    @property
    def _get_height_cal(self) -> HeightCal:
        return HeightCal(settings=self.settings)

    @property
    def _get_height_pred(self) -> HeightPred:
        return HeightPred(settings=self.settings)

    def process(self, inputs: HeightInput) -> HeightOutput:
        # Step 1: Detect Box
        try:
            box_det_out = self._get_box_detector.process(
                inputs=BoxDetectorInput(image=inputs.image),
            )
            logger.info('Box detection completed successfully.')
        except Exception as e:
            logger.exception('Error during Box detection.')
            raise e

        # Step 2: Detect Pose
        try:
            pose_det_out = self._get_pose_detector.process(
                inputs=PoseDetectorInput(img_origin=inputs.image),
            )
            logger.info('Pose detection completed successfully.')
        except Exception as e:
            logger.exception('Error during Pose detection.')
            raise e

        # Step 3: Calculate Height
        try:
            height_cal_out = self._get_height_cal.process(
                inputs=HeightCalInput(
                    landmarks=pose_det_out.pose_landmarks,
                    img_width=pose_det_out.img_width,
                    img_height=pose_det_out.img_height,
                    px_per_cm=box_det_out.pixel_per_cm,
                ),
            )
            logger.info('Height calculation completed successfully.')
        except Exception as e:
            logger.exception('Error during Height calculation.')
            raise e

        # # Step 4: Predict Height
        # try:
        #     height_pred_out = self._get_height_pred.process(
        #         inputs=HeightPredInput(x=height_cal_out.distances),
        #     )
        #     logger.info('Height prediction completed successfully.')
        # except Exception as e:
        #     logger.exception('Error during Height prediction.')
        #     raise e

        # return HeightOutput(results=height_pred_out.pred)

        return HeightOutput(results=height_cal_out.heights)
