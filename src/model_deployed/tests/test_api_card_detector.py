from __future__ import annotations

import os
import unittest

import requests  # type: ignore


class TestCardDetectorAPI(unittest.TestCase):
    def setUp(self) -> None:
        # Đường dẫn tới ảnh mẫu dùng để test
        self.image_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/resource/data/demo_data_card/LuThihSen3.jpg'  # bạn có thể thay đổi
        self.api_url = 'http://localhost:5000/v1/card_detector'

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f'Image not found: {self.image_path}')

    def test_card_detector(self):
        with open(self.image_path, 'rb') as img_file:
            files = {
                'file': ('sample_card.jpg', img_file, 'image/jpeg'),
            }
            response = requests.post(self.api_url, files=files)

        print('Status Code:', response.status_code)
        print('Response JSON:', response.json())

        self.assertEqual(response.status_code, 200)

        json_data = response.json()
        self.assertIn('message', json_data)
        self.assertIn('info', json_data)
        self.assertIn('bboxes', json_data['info'])
        self.assertIsInstance(json_data['info']['bboxes'], list)


if __name__ == '__main__':
    unittest.main()
