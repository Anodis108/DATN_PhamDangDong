from __future__ import annotations

import asyncio
import unittest

import cv2
from common import get_settings
from infrastructure.text_ocr import TextOCRModel
from infrastructure.text_ocr import TextOCRModelInput


class TestTextOCROnly(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = get_settings()
        self.text_ocr = TextOCRModel(settings=self.settings)

    def test_text_ocr(self):
        # Đường dẫn ảnh cần test
        image_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/src/model_deployed/text_detector_preproces.png'
        img = cv2.imread(image_path)
        self.assertIsNotNone(img, 'Ảnh không được load thành công!')

        # Dữ liệu Input từ bạn cung cấp
        class_list = [
            'birth', 'birth', 'birth', 'birth', 'birth',
            'birth', 'birth', 'birth', 'origin', 'birth',
            'birth', 'birth', 'name', 'no', 'birth',
        ]

        bboxes_list = [
            [50.874, 685.8, 272.72, 734.02],
            [450.54, 440.98, 676.28, 508.54],
            [324.76, 375.43, 602.73, 427.54],
            [327.44, 437.88, 680.06, 509.03],
            [468.31, 376.0, 603.44, 425.55],
            [328.11, 587.21, 592.55, 644.0],
            [854.87, 517.65, 1095.7, 574.21],
            [643.12, 369.57, 921.91, 428.23],
            [287.24, 32.348, 1228.4, 126.73],
            [50.993, 649.8, 273.15, 733.26],
            [600.63, 186.71, 789.79, 258.13],
            [326.73, 542.22, 615.89, 643.88],
            [342.65, 515.61, 890.05, 570.6],
            [321.94, 265.27, 824.48, 362.13],
            [353.04, 448.09, 879.38, 523.46],
        ]

        # Chuyển đổi về định dạng phù hợp (float -> int)
        bboxes_list = [list(map(int, box)) for box in bboxes_list]

        # Tạo input cho model OCR
        text_input = TextOCRModelInput(
            img=img,
            class_list=class_list,
            bboxes_list=bboxes_list,
        )

        # Gọi hàm xử lý OCR
        text_output = asyncio.run(self.text_ocr.process(text_input))

        # In kết quả ra console
        for result in text_output.results:
            print('Class:', result['class'])
            print('Bounding box:', result['bounding_box'])
            print('Text:', result['text'])
            print('-' * 50)

        # Vẽ kết quả lên ảnh
        img_output = img.copy()
        for result in text_output.results:
            x1, y1, x2, y2 = map(int, result['bounding_box'])
            cls = result['class']
            text = result['text']

            cv2.rectangle(img_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img_output,
                f'{cls}: {text}',
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Lưu ảnh có kết quả OCR
        cv2.imwrite('text_ocr_output.png', img_output)


if __name__ == '__main__':
    unittest.main()
