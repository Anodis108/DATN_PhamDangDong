from __future__ import annotations

from typing import List
from typing import Tuple

import numpy as np
from common.logs.logs import get_logger
from common.settings import Settings
from mediapipe.tasks.python.vision.pose_landmarker import Landmark

from ..base_cal import CalHeight
from ..base_cal import CalHeightInput
from ..base_cal import CalHeightOutput

logger = get_logger(__name__)


class CalHeight2D(CalHeight):
    settings: Settings

    async def process(self, inputs: CalHeightInput) -> CalHeightOutput:
        heights, distances, cm_direct, cm_sum, diffs = [], [], [], [], []

        for lm in inputs.landmarks:
            h, dist = self.calc_img_height(
                lm=lm,
                img_w=inputs.img_width,
                img_h=inputs.img_height,
                px_per_cm=inputs.px_per_cm,
            )
            cm_dir, cm_s, diff = self.compare_heights(
                height=h,
                distances=dist,
                px_per_cm=inputs.px_per_cm,
            )
            heights.append(h)
            distances.append(dist)
            cm_direct.append(cm_dir)
            cm_sum.append(cm_s)
            diffs.append(diff)

        return CalHeightOutput(
            heights=heights,
            distances=distances,
            cm_direct=cm_direct,
            cm_sum=cm_sum,
            diffs=diffs,
        )

    def calc_img_height(self, lm: List[Landmark], img_w: float, img_h: float, px_per_cm: float) -> Tuple[float, List[float]]:
        ankle_l = lm[27]
        heel_l = lm[29]
        foot_l = lm[31]
        knee_l = lm[25]
        hip_r = lm[24]
        hip_l = lm[23]
        shoulder_r = lm[12]
        shoulder_l = lm[11]
        mouth_r = lm[10]
        mouth_l = lm[9]
        nose = lm[0]

        p_ankle_l = (int(ankle_l.x * img_w), int(ankle_l.y * img_h))
        p_heel_l = (int(heel_l.x * img_w), int(heel_l.y * img_h))
        p_foot_l = (int(foot_l.x * img_w), int(foot_l.y * img_h))
        p_knee_l = (int(knee_l.x * img_w), int(knee_l.y * img_h))
        p_hip_r = (int(hip_r.x * img_w), int(hip_r.y * img_h))
        p_hip_l = (int(hip_l.x * img_w), int(hip_l.y * img_h))
        p_shoulder_r = (int(shoulder_r.x * img_w), int(shoulder_r.y * img_h))
        p_shoulder_l = (int(shoulder_l.x * img_w), int(shoulder_l.y * img_h))
        p_mouth_r = (int(mouth_r.x * img_w), int(mouth_r.y * img_h))
        p_mouth_l = (int(mouth_l.x * img_w), int(mouth_l.y * img_h))
        p_nose = (int(nose.x * img_w), int(nose.y * img_h))

        d_ankle_heel_foot = self.cal_perpendicular_distance(
            p_ankle_l, p_heel_l, p_foot_l,
        )
        d_knee_ankle = self.cal_distance(p_knee_l, p_ankle_l)
        d_hip_knee = self.cal_distance(p_hip_l, p_knee_l)
        d_shoulder_hip = self.cal_distance(
            self.cal_midpoint(p_shoulder_l, p_shoulder_r),
            self.cal_midpoint(p_hip_l, p_hip_r),
        )
        d_mouth_shoulder = self.cal_distance(
            self.cal_midpoint(p_mouth_l, p_mouth_r),
            self.cal_midpoint(p_shoulder_l, p_shoulder_r),
        )
        d_nose_mouth = self.cal_perpendicular_distance(
            p_nose, p_mouth_l, p_mouth_r,
        )
        d_nose_Tophead = 3.236 * d_nose_mouth

        height = (
            d_ankle_heel_foot +
            d_knee_ankle +
            d_hip_knee +
            d_shoulder_hip +
            d_mouth_shoulder +
            d_nose_mouth +
            d_nose_Tophead
        )

        dists = [
            d_ankle_heel_foot,
            d_knee_ankle,
            d_hip_knee,
            d_shoulder_hip,
            d_mouth_shoulder,
            d_nose_mouth,
            d_nose_Tophead,
        ]

        return height / px_per_cm, dists

    # khoảng cách giữa 2 điểm
    def cal_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        # Calculate Euclidean distance between two points
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # điểm trung tuyến
    def cal_midpoint(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> Tuple[float, float]:
        # Calculate midpoint between two points
        return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

    # Tính khoảng cách vuông góc đến đường trung tuyến
    def cal_perpendicular_distance(self, point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
        # Check if the line is vertical (x coordinates are the same)
        if line_end[0] == line_start[0]:
            # If the line is vertical, the perpendicular distance is simply the absolute difference in x coordinates
            return abs(point[0] - line_start[0])
        else:
            # Calculate the equation of the line and then the perpendicular distance
            line_slope = (line_end[1] - line_start[1]) / \
                (line_end[0] - line_start[0])
            line_intercept = line_start[1] - line_slope * line_start[0]
            perpendicular_distance = abs(
                line_slope * point[0] - point[1] + line_intercept,
            ) / (line_slope ** 2 + 1) ** 0.5
            return perpendicular_distance

    def compare_heights(self, height: float, distances: List[float], px_per_cm: float) -> Tuple[float, float, float]:
        cm_dir = height
        cm_s = sum(d / px_per_cm for d in distances)
        diff = abs(cm_dir - cm_s)

        logger.info('=== So sánh chiều cao ===')
        logger.info(f'cm_dir: {cm_dir:.2f} cm')
        logger.info(f'cm_sum: {cm_s:.2f} cm')
        logger.info(f'Chênh lệch: {diff:.4f} cm')
        logger.info('✅ Gần giống' if diff < 1e-2 else '⚠️ Khác nhau')

        return cm_dir, cm_s, diff
