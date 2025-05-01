from __future__ import annotations

import unittest

import cv2
import requests  # type: ignore


class TestTextDetectorAPI(unittest.TestCase):

    def setUp(self) -> None:
        """Cài đặt ban đầu"""
        self.image_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/src/model_deployed/processed_output.jpg'
        self.api_url = 'http://localhost:5000/v1/text_detector'  # URL của API

    def test_text_detector(self):
        """Test API Text Detector"""
        # Đọc ảnh từ file
        loaded_embedding = cv2.imread(self.image_path)
        print('Loaded shape:', loaded_embedding.shape)
        print(type(loaded_embedding))
        # Convert ảnh sang list 3D theo yêu cầu API
        img_list = loaded_embedding.tolist()  # Đây là danh sách 3D
        print(type(img_list))
        # Define bbox (bounding box)

        # Payload với ảnh đã được chuyển sang danh sách 3D và bbox dưới dạng List[float]
        payload = {
            'image': img_list,  # ảnh dưới dạng list 3D
        }

        # Gửi yêu cầu POST đến API
        response = requests.post(self.api_url, json=payload)

        print(f'Status Code: {response.status_code}')
        print('Raw response:', response.text)

        # Kiểm tra mã trạng thái và phản hồi từ API
        self.assertEqual(response.status_code, 200)

        response_json = response.json()
        self.assertIn('message', response_json)
        self.assertEqual(response_json['message'], 'Process successfully !!!')

        self.assertIn('info', response_json)
        self.assertIn('bboxes', response_json['info'])
        self.assertIsInstance(response_json['info']['bboxes'], list)
        self.assertGreater(len(response_json['info']['bboxes']), 0)


if __name__ == '__main__':
    unittest.main()
