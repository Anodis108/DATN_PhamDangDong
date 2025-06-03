from __future__ import annotations

import json
import unittest

import cv2
from common import get_settings
from infrastructure.height_predictor import HeightPredictorModel
from infrastructure.height_predictor import HeightPredictorModelInput



class TestHeightPred(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = get_settings()
        self.height_predictor_model = HeightPredictorModel.get_service(settings=self.settings)

    def test_height_pred(self):

        distance = [
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0
            ]
        ]

        inputs = HeightPredictorModelInput(x=distance)

        outputs = self.height_predictor_model.process(inputs)

        # Chuẩn bị dữ liệu đầu ra dưới dạng dictionary
        result = {
            'pred': outputs.pred
        }

        # In ra kết quả dưới dạng JSON
        print(json.dumps(result, indent=4))


if __name__ == '__main__':
    unittest.main()
