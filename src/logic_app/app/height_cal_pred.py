from __future__ import annotations

from pathlib import Path

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
from service.draw import VisualizationInput
from service.draw import VisualizationService
from service.write_csv import CSVWriterInput
from service.write_csv import CSVWriterService


logger = get_logger(__name__)


class HeightInput(BaseModel):
    image: np.ndarray
    img_name: str


class HeightOutput(BaseModel):
    # status: bool
    results: list[float]
    out_path: str


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

    @property
    def _get_draw(self) -> VisualizationService:
        return VisualizationService(settings=self.settings)

    @property
    def _get_write_csv(self) -> CSVWriterService:
        return CSVWriterService(settings=self.settings)

    async def process(self, inputs: HeightInput) -> HeightOutput:
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

        # Step 3.1: draw
        try:
            height_pre = height_cal_out.heights
            # Trích xuất thông tin từ tên file
            pose_num, height_truth, name_no_ext = self.parse_height_input_from_img_name(
                inputs.img_name,
            )

            draw_out = self._get_draw.process(
                inputs=VisualizationInput(
                    bboxes=np.array(box_det_out.bboxes),
                    confidences=np.array(box_det_out.scores),
                    height_cm=height_pre,   # Sẽ cần sửa trong tương lai
                    image=inputs.image,
                    name_image=name_no_ext,
                    pose_landmarks_list=pose_det_out.pose_landmarks,
                ),
            )
            logger.info('Draw completed successfully.')
        except Exception as e:
            logger.exception('Error during Draw.')
            raise e

        # Step 3.2: write csv
        try:
            write_csv_out = await self._get_write_csv.process(
                inputs=CSVWriterInput(
                    distances=height_cal_out.distances,
                    height_truth=height_truth,
                    pose_landmarks_list=pose_det_out.pose_landmarks,
                    pose_num=pose_num,
                    height_pre=height_pre,
                    px_per_cm=box_det_out.pixel_per_cm,
                ),
            )
            logger.info(
                f'✅ Write CSV completed successfully.{write_csv_out}',
                extra={
                    'landmarks_csv_written': write_csv_out.landmarks_csv_written,
                    'distances_csv_written': write_csv_out.distances_csv_written,
                },
            )
        except Exception as e:
            logger.exception('Error during Write csv.')
            raise e
        logger.info(f'✅ pixcel per cm {box_det_out.pixel_per_cm}')
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

        return HeightOutput(
            results=height_cal_out.heights,
            out_path=draw_out.output_path,
        )

    def parse_height_input_from_img_name(self, img_name: str) -> tuple[int, float, str]:
        """
        Trích xuất pose_num và height_truth từ tên file, ví dụ:
        '1_DungThang_Base_1_170.jpg' → (pose_num=1, height_truth=170.0)
        """
        name_image = Path(img_name).stem
        name_no_ext = Path(name_image).stem  # loại bỏ .jpg
        parts = name_no_ext.split('_')

        try:
            pose_num = int(parts[0])
            height_truth = float(parts[-1])
        except (IndexError, ValueError):
            raise ValueError(f'Không thể phân tích tên file: {img_name}')

        return pose_num, height_truth, name_no_ext
