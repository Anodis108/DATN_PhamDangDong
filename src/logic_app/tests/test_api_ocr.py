from __future__ import annotations

import json
import os
import unittest

import requests  # type: ignore


class TestOCRAPI(unittest.TestCase):
    def setUp(self) -> None:
        # Đường dẫn tới ảnh mẫu dùng để test
        self.image_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/resource/data/demo_data_card/LuThihSen3.jpg'  # bạn có thể thay đổi
        self.api_url = 'http://localhost:5001/v1/ocr'  # Thay đổi URL API của bạn

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f'Image not found: {self.image_path}')

    def test_ocr(self):
        # Đọc ảnh và gửi yêu cầu POST tới API OCR
        with open(self.image_path, 'rb') as img_file:
            files = {
                'file': ('sample_image.jpg', img_file, 'image/jpeg'),
            }
            response = requests.post(self.api_url, files=files)

        # In ra mã trạng thái và JSON phản hồi với format đẹp
        print('Status Code:', response.status_code)

        try:
            json_data = response.json()
            print('Response JSON:')
            # In đẹp, giữ nguyên tiếng Việt nếu có
            print(json.dumps(json_data, indent=4, ensure_ascii=False))

            # Kiểm tra mã trạng thái và nội dung
            self.assertEqual(response.status_code, 200)
            self.assertIn('message', json_data)
            self.assertEqual(json_data['message'], 'Process successfully !!!')
        except Exception as e:
            print('Failed to parse JSON response:', e)
            print('Raw Response:', response.text)
            self.fail('Invalid JSON response received.')
        # self.assertIsNotNone(json_data['info'])
        # self.assertIn('info_text', json_data['info'])
        # self.assertGreater(len(json_data['info']['info_text']), 0)


if __name__ == '__main__':
    unittest.main()
