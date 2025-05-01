from __future__ import annotations

import unittest

import cv2
import requests  # type: ignore


class TestTextOCRAPI(unittest.TestCase):

    def setUp(self) -> None:
        """Cài đặt ban đầu"""
        self.image_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/src/model_deployed/processed_output.jpg'
        self.api_url = 'http://localhost:5000/v1/text_ocr'  # Thay đổi nếu API khác

        # Bboxes, classes và confs từ kết quả thực tế
        self.bboxes = [
            [443.92, 440.61, 676.26, 508.54],
            [50.83, 685.97, 272.59, 734.06],
            [324.81, 375.63, 602.99, 427.21],
            [328.44, 437.50, 680.15, 509.05],
            [468.20, 376.12, 603.64, 425.35],
            [328.26, 587.16, 592.28, 643.93],
            [646.16, 370.60, 922.66, 429.67],
            [848.45, 517.72, 1095.62, 574.21],
            [475.71, 449.05, 683.80, 503.54],
            [286.65, 32.61, 1228.35, 126.43],
            [600.45, 186.83, 789.78, 258.03],
            [50.85, 652.01, 273.06, 733.51],
            [326.94, 540.66, 617.33, 643.51],
            [343.97, 515.60, 889.62, 570.29],
            [321.96, 265.16, 823.43, 362.08],
        ]
        self.classes = [
            'birth', 'birth', 'birth', 'birth', 'birth', 'birth',
            'birth', 'birth', 'birth', 'origin', 'birth', 'birth',
            'birth', 'name', 'no',
        ]

    def test_text_ocr_with_real_data(self):
        """Test Text OCR API với dữ liệu bounding box thực tế"""
        img = cv2.imread(self.image_path).tolist()

        payload = {
            'img': img,
            'bboxes': self.bboxes,
            'classes': self.classes,
        }

        response = requests.post(self.api_url, json=payload)

        print(f'Status Code: {response.status_code}')
        print('Raw response:', response.text)

        self.assertEqual(response.status_code, 200)

        response_json = response.json()

        self.assertIn('info', response_json)
        info = response_json['info']['info_text']
        self.assertIsInstance(info, list)
        self.assertEqual(len(info), len(self.bboxes))

        # In kết quả nhận dạng từng vùng
        for i, item in enumerate(info):
            self.assertIn('class_name', item)
            self.assertIn('bounding_box', item)
            self.assertIn('text', item)

            print(
                f"[{item['class_name']}] → {item['text']} (box: {item['bounding_box']})",
            )


if __name__ == '__main__':
    unittest.main()
