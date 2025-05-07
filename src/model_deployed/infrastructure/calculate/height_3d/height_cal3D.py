from __future__ import annotations

from typing import List
from typing import Tuple

import numpy as np
from common.logs.logs import get_logger
from common.settings import Settings
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from ..base_cal import CalHeight
from ..base_cal import CalHeightInput
from ..base_cal import CalHeightOutput

logger = get_logger(__name__)


class CalHeight3D(CalHeight):
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
            # đang hơi thừa vì ouput heights có / px_per_cm rồi
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

    def calc_img_height(self, lm: List[NormalizedLandmark], img_w: float, img_h: float, px_per_cm: float) -> Tuple[float, List[float]]:
        left_ankle = lm[27]
        left_heel = lm[29]
        left_foot_index = lm[31]
        left_knee = lm[25]
        right_hip = lm[24]
        left_hip = lm[23]
        right_shoulder = lm[12]
        left_shoulder = lm[11]
        right_mouth = lm[10]
        left_mouth = lm[9]
        nose = lm[0]

        # Convert normalized landmark coordinates to pixel coordinates
        px_left_ankle = (int(left_ankle.x * img_w), int(left_ankle.y * img_h))
        px_left_heel = (int(left_heel.x * img_w), int(left_heel.y * img_h))
        px_left_foot_index = (
            int(left_foot_index.x * img_w),
            int(left_foot_index.y * img_h),
        )
        px_left_knee = (int(left_knee.x * img_w), int(left_knee.y * img_h))
        px_right_hip = (int(right_hip.x * img_w), int(right_hip.y * img_h))
        px_left_hip = (int(left_hip.x * img_w), int(left_hip.y * img_h))
        px_right_shoulder = (
            int(right_shoulder.x * img_w),
            int(right_shoulder.y * img_h),
        )
        px_left_shoulder = (
            int(left_shoulder.x * img_w),
            int(left_shoulder.y * img_h),
        )
        px_right_mouth = (
            int(right_mouth.x * img_w),
            int(right_mouth.y * img_h),
        )
        px_left_mouth = (int(left_mouth.x * img_w), int(left_mouth.y * img_h))
        px_nose = (int(nose.x * img_w), int(nose.y * img_h))

        # Extract 3D coordinates and 2D coordinates (ignoring Z)
        left_ankle_3d = (lm[27].x, lm[27].y, lm[27].z)
        left_heel_3d = (lm[29].x, lm[29].y, lm[29].z)
        left_foot_index_3d = (lm[31].x, lm[31].y, lm[31].z)
        left_knee_3d = (lm[25].x, lm[25].y, lm[25].z)
        right_hip_3d = (lm[24].x, lm[24].y, lm[24].z)
        left_hip_3d = (lm[23].x, lm[23].y, lm[23].z)
        right_shoulder_3d = (lm[12].x, lm[12].y, lm[12].z)
        left_shoulder_3d = (lm[11].x, lm[11].y, lm[11].z)
        right_mouth_3d = (lm[10].x, lm[10].y, lm[10].z)
        left_mouth_3d = (lm[9].x, lm[9].y, lm[9].z)
        nose_3d = (lm[0].x, lm[0].y, lm[0].z)

        left_ankle_2d = (lm[27].x, lm[27].y)
        left_heel_2d = (lm[29].x, lm[29].y)
        left_foot_index_2d = (lm[31].x, lm[31].y)
        left_knee_2d = (lm[25].x, lm[25].y)
        right_hip_2d = (lm[24].x, lm[24].y)
        left_hip_2d = (lm[23].x, lm[23].y)
        right_shoulder_2d = (lm[12].x, lm[12].y)
        left_shoulder_2d = (lm[11].x, lm[11].y)
        right_mouth_2d = (lm[10].x, lm[10].y)
        left_mouth_2d = (lm[9].x, lm[9].y)
        nose_2d = (lm[0].x, lm[0].y)

        # self.cal distances using pixel coordinates
        dis_ankle_heel_foot_index = self.cal_perpendicular_distance(
            px_left_ankle, px_left_heel,
            px_left_foot_index,
        )
        dis_knee_ankle = self.cal_distance(px_left_knee, px_left_ankle)
        dis_hip_knee = self.cal_distance(px_left_hip, px_left_knee)
        dis_midpoint_shoulder_hip = self.cal_distance(
            self.cal_midpoint(px_left_shoulder, px_right_shoulder),
            self.cal_midpoint(px_left_hip, px_right_hip),
        )
        # dis_midpoint_shoulder_nose = self.cal_distance(self.cal_midpoint(px_left_shoulder, px_right_shoulder), px_nose)
        dis_midpoint_mouth_shoulder = self.cal_distance(
            self.cal_midpoint(px_left_mouth, px_right_mouth),
            self.cal_midpoint(px_left_shoulder, px_right_shoulder),
        )
        dis_nose_mouth = self.cal_perpendicular_distance(
            px_nose, px_left_mouth, px_right_mouth,
        )
        # 0.5  # Assuming 0.5 as the ratio for simplicity, adjust as needed
        dis_nose_top_of_head = 3.236 * dis_nose_mouth

        # self.cal distances using 3D coordinates
        dis_ankle_heel_foot_index_3d = self.cal_perpendicular_distance3D(
            left_ankle_3d, left_heel_3d,
            left_foot_index_3d,
        )
        dis_knee_ankle_3d = self.cal_distance3D(left_knee_3d, left_ankle_3d)
        dis_hip_knee_3d = self.cal_distance3D(left_hip_3d, left_knee_3d)
        dis_midpoint_shoulder_hip_3d = self.cal_distance3D(
            self.cal_midpoint3D(left_shoulder_3d, right_shoulder_3d),
            self.cal_midpoint3D(left_hip_3d, right_hip_3d),
        )
        dis_midpoint_mouth_shoulder_3d = self.cal_distance3D(
            self.cal_midpoint3D(left_mouth_3d, right_mouth_3d),
            self.cal_midpoint3D(left_shoulder_3d, right_shoulder_3d),
        )
        dis_nose_mouth_3d = self.cal_perpendicular_distance3D(
            nose_3d, left_mouth_3d, right_mouth_3d,
        )
        # Assuming a ratio, adjust as needed
        dis_nose_top_of_head_3d = 3.236 * dis_nose_mouth_3d

        # self.cal distances using 2D coordinates
        dis_ankle_heel_foot_index_2d = self.cal_perpendicular_distance(
            left_ankle_2d, left_heel_2d, left_foot_index_2d,
        )
        dis_knee_ankle_2d = self.cal_distance(left_knee_2d, left_ankle_2d)
        dis_hip_knee_2d = self.cal_distance(left_hip_2d, left_knee_2d)
        dis_midpoint_shoulder_hip_2d = self.cal_distance(
            self.cal_midpoint(left_shoulder_2d, right_shoulder_2d),
            self.cal_midpoint(left_hip_2d, right_hip_2d),
        )
        dis_midpoint_mouth_shoulder_2d = self.cal_distance(
            self.cal_midpoint(left_mouth_2d, right_mouth_2d),
            self.cal_midpoint(left_shoulder_2d, right_shoulder_2d),
        )
        dis_nose_mouth_2d = self.cal_perpendicular_distance(
            nose_2d, left_mouth_2d, right_mouth_2d,
        )
        # Assuming a ratio, adjust as needed
        dis_nose_top_of_head_2d = 3.236 * dis_nose_mouth_2d

        # Regression coefficients
        rc1 = 1
        rc2 = 1
        rc3 = 1
        rc4 = 1
        rc5 = 1
        rc6 = 1
        rc7 = 1

        d1_ = rc1 * (dis_ankle_heel_foot_index / px_per_cm)
        d2_ = rc2 * (dis_knee_ankle / px_per_cm)
        d3_ = rc3 * (dis_hip_knee / px_per_cm)
        d4_ = rc4 * (dis_midpoint_shoulder_hip / px_per_cm)
        d5_ = rc5 * (dis_midpoint_mouth_shoulder / px_per_cm)
        d6_ = rc6 * (dis_nose_mouth / px_per_cm)
        d7_ = rc7 * (dis_nose_top_of_head / px_per_cm)

        d1_2d = rc1 * dis_ankle_heel_foot_index_2d
        d2_2d = rc2 * dis_knee_ankle_2d
        d3_2d = rc3 * dis_hip_knee_2d
        d4_2d = rc4 * dis_midpoint_shoulder_hip_2d
        d5_2d = rc5 * dis_midpoint_mouth_shoulder_2d
        d6_2d = rc6 * dis_nose_mouth_2d
        d7_2d = rc7 * dis_nose_top_of_head_2d

        d1_3d = rc1 * dis_ankle_heel_foot_index_3d
        d2_3d = rc2 * dis_knee_ankle_3d
        d3_3d = rc3 * dis_hip_knee_3d
        d4_3d = rc4 * dis_midpoint_shoulder_hip_3d
        d5_3d = rc5 * dis_midpoint_mouth_shoulder_3d
        d6_3d = rc6 * dis_nose_mouth_3d
        d7_3d = rc7 * dis_nose_top_of_head_3d

        k1 = d1_3d / d1_2d
        k2 = d2_3d / d2_2d
        k3 = d3_3d / d3_2d
        k4 = d4_3d / d4_2d
        k5 = d5_3d / d5_2d
        k6 = d6_3d / d6_2d
        k7 = d7_3d / d7_2d

        d1 = d1_ * k1
        d2 = d2_ * k2
        d3 = d3_ * k3
        d4 = d4_ * k4
        d5 = d5_ * k5
        d6 = d6_ * k6
        d7 = d7_ * k7

        # Sum up to self.cal the height

        heightf = (
            d1 +
            d2 +
            d3 +
            d4 +
            d5 +
            d6 +
            d7 +
            0
        )

        # heightf_2d = (
        #     d1_2d +
        #     d2_2d +
        #     d3_2d +
        #     d4_2d +
        #     d5_2d +
        #     d6_2d +
        #     d7_2d +
        #     0
        # )

        # heightf_3d = (
        #     d1_3d +
        #     d2_3d +
        #     d3_3d +
        #     d4_3d +
        #     d5_3d +
        #     d6_3d +
        #     d7_3d +
        #     0
        # )

        # Store distances in a list
        distances = [
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
        ]
        # print(k1,k2,k3,k4,k5,k6,k7)
        # print("Height: ", heightf)

        return heightf, distances

    # khoảng cách giữa 2 điểm
    def cal_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        # self.cal Euclidean distance between two points
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # điểm trung tuyến
    def cal_midpoint(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> Tuple[float, float]:
        # self.cal midpoint between two points
        return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

    # Tính khoảng cách vuông góc đến đường trung tuyến
    def cal_perpendicular_distance(self, point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
        # Check if the line is vertical (x coordinates are the same)
        if line_end[0] == line_start[0]:
            # If the line is vertical, the perpendicular distance is simply the absolute difference in x coordinates
            return abs(point[0] - line_start[0])
        else:
            # self.cal the equation of the line and then the perpendicular distance
            line_slope = (line_end[1] - line_start[1]) / \
                (line_end[0] - line_start[0])
            line_intercept = line_start[1] - line_slope * line_start[0]
            perpendicular_distance = abs(
                line_slope * point[0] - point[1] + line_intercept,
            ) / (line_slope ** 2 + 1) ** 0.5
            return perpendicular_distance

    def cal_distance3D(self, point1, point2):
        # self.cal Euclidean distance between two points
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def cal_midpoint3D(self, point1, point2):
        # self.cal midpoint between two points
        return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2, (point1[2] + point2[2]) / 2)

    def cal_perpendicular_distance3D(self, point, line_start, line_end):
        # self.cal the perpendicular distance from a point to a line defined by two points in 3D space
        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        nearest = line_unitvec * t
        distance = np.linalg.norm(point_vec - nearest * line_len)
        return distance

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
