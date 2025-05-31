from __future__ import annotations

from functools import cached_property
from typing import Tuple

import numpy as np
from common.bases import BaseModel
from common.bases import BaseService
from common.logs.logs import get_logger
from common.settings import Settings
from ultralytics import YOLO

logger = get_logger(__name__)


class BoxDetectorModelInput(BaseModel):
    img: np.ndarray


class BoxDetectorModelOutput(BaseModel):
    bboxes: np.ndarray  # (N, 4)
    scores: np.ndarray  # (N,)
    pixel_per_cm: float


class BoxDetectorModel(BaseService):
    settings: Settings

    @cached_property
    def model_loaded(self) -> YOLO:
        return YOLO(self.settings.box_detector.model_path)

    async def process(self, inputs: BoxDetectorModelInput) -> BoxDetectorModelOutput:
        scores, bboxes, pixel_per_cm = self.forward(
            inputs.img, self.settings.box_detector.conf,
        )
        return BoxDetectorModelOutput(bboxes=bboxes, scores=scores, pixel_per_cm=pixel_per_cm)

    def forward(self, img: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Performs a forward pass on the Box detection model to extract all bounding boxes and confidence scores,
        then applies NMS and sorts them in descending order of score.

        Returns:
            scores: np.ndarray of shape (N,)
            bboxes_xyxy: np.ndarray of shape (N, 4)
            pixel_per_cm: float - calculated using the height of the best bounding box
        """
        model = self.model_loaded
        results = model(img)[0]

        det_list = []
        h_list = []

        for box in results.boxes:
            score = box.conf[0].item()
            if score < threshold:
                continue
            bbox_xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            bbox_xywh = box.xywh[0].cpu().numpy()  # [x, y, w, h]
            det_list.append(np.append(bbox_xyxy, score))
            h_list.append(bbox_xywh[3])  # h = bbox_xywh[3]

        if not det_list:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
                0.0,
            )

        det_array = np.array(det_list, dtype=np.float32).reshape(-1, 5)
        h_array = np.array(h_list, dtype=np.float32)

        # Apply NMS
        keep_indices = self.nms(det_array)
        det_nms = det_array[keep_indices]
        h_nms = h_array[keep_indices]

        # Sort by score descending
        sorted_indices = np.argsort(det_nms[:, 4])[::-1]
        sorted_det = det_nms[sorted_indices]
        sorted_h = h_nms[sorted_indices]

        scores = sorted_det[:, 4]
        bboxes_xyxy = sorted_det[:, :4]
        best_h = sorted_h[0]  # height of best bbox

        pixel_per_cm = self.cal_pixel_per_cm(best_h)

        return scores, bboxes_xyxy, pixel_per_cm

    def nms(self, dets: np.ndarray) -> list[int]:
        thresh = self.settings.box_detector.conf
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def cal_pixel_per_cm(self, h_box_det: float) -> float:
        return h_box_det / self.settings.box_detector.base_h
