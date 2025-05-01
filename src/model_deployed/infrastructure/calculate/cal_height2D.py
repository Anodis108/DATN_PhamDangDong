from __future__ import annotations

from typing import List
from typing import Tuple

import numpy as np
from common.bases import BaseModel
from common.bases import BaseService
from common.logs.logs import get_logger
from common.settings import Settings
from mediapipe.tasks.python.vision.pose_landmarker import Landmark

logger = get_logger(__name__)


class HeightInput(BaseModel):
    landmarks: List[List[Landmark]]
    img_width: float
    img_height: float
    px_per_cm: float

class HeightOutput(BaseModel):
    heights: List[float]
    distances: List[List[float]]
    cm_direct: List[float]
    cm_sum: List[float]
    diffs: List[float]

class HeightService(BaseService):
    settings: Settings

    async def process(self, input: HeightInput) -> HeightOutput:
        heights, distances, cm_direct, cm_sum, diffs = [], [], [], [], []

        for lm in input.landmarks:
            h, dist = self.calc_img_height(
                lm=lm,
                w=input.img_width,
                h=input.img_height,
            )
            cm_dir, cm_s, diff = self.compare_heights(
                height=h,
                distances=dist,
                px_per_cm=input.px_per_cm
            )
            heights.append(h)
            distances.append(dist)
            cm_direct.append(cm_dir)
            cm_sum.append(cm_s)
            diffs.append(diff)

        return HeightOutput(
            heights=heights,
            distances=distances,
            cm_direct=cm_direct,
            cm_sum=cm_sum,
            diffs=diffs,
        )
        
    
    def calc_img_height(self, lm: List[Landmark], w: float, h: float) -> Tuple[float, List[float]]:
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

        p_ankle_l = (int(ankle_l.x * w), int(ankle_l.y * h))
        p_heel_l = (int(heel_l.x * w), int(heel_l.y * h))
        p_foot_l = (int(foot_l.x * w), int(foot_l.y * h))
        p_knee_l = (int(knee_l.x * w), int(knee_l.y * h))
        p_hip_r = (int(hip_r.x * w), int(hip_r.y * h))
        p_hip_l = (int(hip_l.x * w), int(hip_l.y * h))
        p_shoulder_r = (int(shoulder_r.x * w), int(shoulder_r.y * h))
        p_shoulder_l = (int(shoulder_l.x * w), int(shoulder_l.y * h))
        p_mouth_r = (int(mouth_r.x * w), int(mouth_r.y * h))
        p_mouth_l = (int(mouth_l.x * w), int(mouth_l.y * h))
        p_nose = (int(nose.x * w), int(nose.y * h))

        d_ankle_heel = self.perp_dist(p_ankle_l, p_heel_l, p_foot_l)
        d_knee_ankle = self.dist(p_knee_l, p_ankle_l)
        d_hip_knee = self.dist(p_hip_l, p_knee_l)
        d_shoulder_hip = self.dist(
            self.midpoint(p_shoulder_l, p_shoulder_r),
            self.midpoint(p_hip_l, p_hip_r)
        )
        d_mouth_shoulder = self.dist(
            self.midpoint(p_mouth_l, p_mouth_r),
            self.midpoint(p_shoulder_l, p_shoulder_r)
        )
        d_nose_mouth = self.perp_dist(p_nose, p_mouth_l, p_mouth_r)
        d_nose_head = 3.236 * d_nose_mouth

        height = sum([
            d_ankle_heel,
            d_knee_ankle,
            d_hip_knee,
            d_shoulder_hip,
            d_mouth_shoulder,
            d_nose_mouth,
            d_nose_head
        ])

        dists = [
            d_ankle_heel,
            d_knee_ankle,
            d_hip_knee,
            d_shoulder_hip,
            d_mouth_shoulder,
            d_nose_mouth,
            d_nose_head
        ]

        return height, dists
    
    # khoảng cách giữa 2 điểm
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        # Calculate Euclidean distance between two points
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # điểm trung tuyến
    def calculate_midpoint(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> Tuple[float, float]:
        # Calculate midpoint between two points
        return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

    # Tính khoảng cách vuông góc đến đường trung tuyến 
    def calculate_perpendicular_distance(self, point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
        # Check if the line is vertical (x coordinates are the same)
        if line_end[0] == line_start[0]:
            # If the line is vertical, the perpendicular distance is simply the absolute difference in x coordinates
            return abs(point[0] - line_start[0])
        else:
            # Calculate the equation of the line and then the perpendicular distance
            line_slope = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
            line_intercept = line_start[1] - line_slope * line_start[0]
            perpendicular_distance = abs(line_slope * point[0] - point[1] + line_intercept) / (line_slope ** 2 + 1) ** 0.5
            return perpendicular_distance

    def compare_heights(self, height: float, distances: List[float], px_per_cm: float) -> Tuple[float, float, float]:
        cm_dir = height / px_per_cm
        cm_s = sum(d / px_per_cm for d in distances)
        diff = abs(cm_dir - cm_s)

        logger.info("=== So sánh chiều cao ===")
        logger.info(f"cm_dir: {cm_dir:.2f} cm")
        logger.info(f"cm_sum: {cm_s:.2f} cm")
        logger.info(f"Chênh lệch: {diff:.4f} cm")
        logger.info("✅ Gần giống" if diff < 1e-2 else "⚠️ Khác nhau")

        return cm_dir, cm_s, diff