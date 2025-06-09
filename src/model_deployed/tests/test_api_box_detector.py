from __future__ import annotations

import json  # Thêm để định dạng JSON
import os
import unittest

import requests  # type: ignore


class TestCardDetectorAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.image_path = '/mnt/d/project/DATN/DATN_PhamDangDong/resource/data/data_test/z6194149941298_733dfe9d1f76e5f775c0f9336c4b3f3e.jpg'
        self.api_url = 'http://localhost:5000/v1/box_detector'

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f'Image not found: {self.image_path}')

    def test_card_detector_api(self):
        with open(self.image_path, 'rb') as img_file:
            files = {
                'file': ('sample_card.jpg', img_file, 'image/jpeg'),
            }
            response = requests.post(self.api_url, files=files)

        print('Status Code:', response.status_code)

        # In JSON ra terminal theo định dạng đẹp
        print('Response JSON:')
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))

        # Các kiểm tra
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn('message', json_data)
        self.assertIn('info', json_data)
        self.assertIn('bboxes', json_data['info'])
        self.assertIsInstance(json_data['info']['bboxes'], list)


if __name__ == '__main__':
    unittest.main()
