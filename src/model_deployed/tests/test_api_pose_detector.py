from __future__ import annotations

import json
import os
import unittest

import requests  # type: ignore


class TestPoseDetectorAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.image_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_PhamDangDong/DATN_PhamDangDong/resource/data/data_test/z6194149941298_733dfe9d1f76e5f775c0f9336c4b3f3e.jpg'
        self.api_url = 'http://localhost:5000/v1/pose_detector'

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f'Image not found: {self.image_path}')

    def test_pose_detector_api(self):
        with open(self.image_path, 'rb') as img_file:
            files = {
                'file': ('pose_image.jpg', img_file, 'image/jpeg'),
            }
            response = requests.post(self.api_url, files=files)

        print('Status Code:', response.status_code)
        print('Response Text:', response.text)

        # Kiểm tra trạng thái HTTP trước
        self.assertEqual(
            response.status_code, 200,
            f'API returned status code {response.status_code}: {response.text}',
        )

        # Kiểm tra phản hồi JSON
        try:
            json_data = response.json()
            print('Response JSON:')
            print(json.dumps(json_data, indent=4, ensure_ascii=False))
        except json.JSONDecodeError:
            self.fail(f'Response is not a valid JSON: {response.text}')


if __name__ == '__main__':
    unittest.main()
