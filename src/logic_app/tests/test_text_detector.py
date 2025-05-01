from __future__ import annotations

import unittest

import cv2
from common.utils import get_settings
from fastapi.encoders import jsonable_encoder
from infrastructure.text_detector import TextDetector
from infrastructure.text_detector import TextDetectorInput


class TestTextDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = get_settings()
        self.text_detector = TextDetector(settings=self.settings)
        self.img_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/resource/data/demo_data_card/LuThihSen3.jpg'

    def test_api_text_detection(self):
        test_image = cv2.imread(self.img_path)
        print('Image shape:', test_image.shape, 'dtype:', test_image.dtype)

        dummy_bbox = [
            341.795654296875, 896.8245849609375,
            1506.250732421875, 1659.7244873046875,
        ]
        inputs = TextDetectorInput(img_origin=test_image, bbox=dummy_bbox)

        result = self.text_detector.process(inputs=inputs)

        print(jsonable_encoder(result))

        print('bboxes_list', result.bboxes_list)
        print('class_list', result.class_list)
        print('conf_list', result.conf_list)
        # # Kiểm tra các trường cần thiết có tồn tại
        # self.assertIn('class_list', result.dict())
        # self.assertIn('bboxes_list', result.dict())
        # self.assertIn('conf_list', result.dict())
        # self.assertIn('processed_image', result.dict())

        # # Kiểm tra kiểu dữ liệu
        # self.assertIsInstance(result.class_list, list)
        # self.assertIsInstance(result.bboxes_list, list)
        # self.assertIsInstance(result.conf_list, list)
        # self.assertIsInstance(result.processed_image, list)

        # for conf in result.conf_list:
        #     self.assertIsInstance(conf, float)
