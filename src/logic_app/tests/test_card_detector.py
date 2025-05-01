from __future__ import annotations

import unittest

import cv2
from common.utils import get_settings
from fastapi.encoders import jsonable_encoder
from infrastructure.card_detector import CardDetector
from infrastructure.card_detector import CardDetectorInput


class TestCardDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = get_settings()
        self.card_detector = CardDetector(settings=self.settings)
        self.img_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/resource/data/demo_data_card/LuThihSen3.jpg'

    def test_api_card_detection(self):
        test_image = cv2.imread(self.img_path)
        print(
            'Image shape:', test_image,
            'dtype:', test_image.dtype,
        )

        inputs = CardDetectorInput(image=test_image)
        result = self.card_detector.process(inputs=inputs)
        print(jsonable_encoder(result))

        # Kiểm tra kết quả trả về (có thể là bboxes và scores)
        self.assertIn('bboxes', result.dict())
        self.assertIn('scores', result.dict())
        self.assertIsInstance(result.bboxes, list)
        self.assertIsInstance(result.scores, list)

        # Kiểm tra kiểu dữ liệu của scores (phải là danh sách số thực)
        for score in result.scores:
            self.assertIsInstance(score, float)
