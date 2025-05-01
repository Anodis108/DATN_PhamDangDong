from __future__ import annotations

import unittest

import cv2
from common.utils import get_settings
from fastapi.encoders import jsonable_encoder
from infrastructure.text_ocr import TextOCR
from infrastructure.text_ocr import TextOCRInput


class TestTextOCR(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = get_settings()
        self.text_ocr = TextOCR(settings=self.settings)
        self.img_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/src/model_deployed/processed_output.jpg'

        self.bboxes_list = [
            [443.92, 440.61, 676.26, 508.55],
            [50.83, 685.98, 272.60, 734.06],
            [324.82, 375.64, 602.99, 427.21],
            [328.45, 437.50, 680.15, 509.06],
            [468.21, 376.13, 603.65, 425.35],
            [328.27, 587.17, 592.29, 643.94],
            [646.16, 370.61, 922.67, 429.67],
            [848.46, 517.72, 1095.63, 574.21],
            [475.71, 449.05, 683.80, 503.54],
            [286.65, 32.61, 1228.35, 126.44],
            [600.45, 186.84, 789.79, 258.04],
            [50.85, 652.01, 273.07, 733.51],
            [326.94, 540.66, 617.33, 643.52],
            [343.97, 515.61, 889.62, 570.29],
            [321.96, 265.17, 823.43, 362.08],
        ]

        self.class_list = [
            'birth', 'birth', 'birth', 'birth', 'birth', 'birth', 'birth', 'birth',
            'birth', 'origin', 'birth', 'birth', 'birth', 'name', 'no',
        ]

    def test_api_text_ocr(self):
        test_image = cv2.imread(self.img_path)
        print('Image shape:', test_image.shape, 'dtype:', test_image.dtype)

        inputs = TextOCRInput(
            img=test_image,
            class_list=self.class_list,
            bboxes_list=self.bboxes_list,
        )

        result = self.text_ocr.process(inputs=inputs)

        # Print kết quả JSON
        print(jsonable_encoder(result))

        # Nếu result có dạng list dicts
        print('\n--- RESULTS ---')
        # for i, r in enumerate(result.results):
        #     print(f'[{i}] Text:', r)

        print('Số lượng kết quả:', len(result.results))


if __name__ == '__main__':
    unittest.main()
