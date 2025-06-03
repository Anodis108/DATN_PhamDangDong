from __future__ import annotations

import asyncio
import json
import unittest

import cv2
from common import get_settings
from infrastructure.pose_detector import PoseDetectorModel
from infrastructure.pose_detector import PoseDetectorModelInput


class TestPoseDetector(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = get_settings()  # Giả sử settings được tải từ một hàm get_settings()
        self.pose_detector_model = PoseDetectorModel(settings=self.settings)

    def test_pose_detector(self):
        # Thay thế bằng đường dẫn ảnh của bạn
        image_path = '/mnt/d/project/DATN/DATN_PhamDangDong/resource/data/data_test/z6194149941298_733dfe9d1f76e5f775c0f9336c4b3f3e.jpg'
        img = cv2.imread(image_path)
        self.assertIsNotNone(
            img, 'Không thể đọc được ảnh từ đường dẫn cung cấp',
        )

        # Kiểm tra định dạng ảnh
        b, g, r = img[0, 0]
        if b != r:
            print('Ảnh đang ở định dạng BGR')
        else:
            print('Ảnh có thể là RGB hoặc ảnh đơn sắc')

        # Tạo đối tượng input cho model
        inputs = PoseDetectorModelInput(img=img)

        # Kiểm tra phương thức process (sử dụng asyncio để gọi hàm bất đồng bộ)
        outputs = asyncio.run(self.pose_detector_model.process(inputs))

        # Chuẩn bị dữ liệu đầu ra dưới dạng dictionary để in ra JSON
        result = {
            'pose_landmarks': outputs.pose_landmarks,
            'img_h': outputs.img_height,
            'img_w': outputs.img_width,
        }

        # In ra kết quả dưới dạng JSON
        print(json.dumps(result, indent=4))

        # Kiểm tra nếu có landmarks và in thông tin
        if len(outputs.pose_landmarks) > 0:
            print(
                f'Image dimensions: {outputs.img_width:.1f}x{outputs.img_height:.1f} pixels',
            )
            for person_landmarks in outputs.pose_landmarks:
                for landmark in person_landmarks:
                    # In ra thông tin landmarks
                    print(
                        f'Landmark: x={landmark["x"]:.3f}, y={landmark["y"]:.3f}, z={landmark["z"]:.3f}',
                    )
        else:
            print('Không phát hiện landmarks nào.')

        # Vẽ landmarks lên ảnh
        for person_landmarks in outputs.pose_landmarks:
            for landmark in person_landmarks:
                # Quy đổi to pixel
                x, y = int(
                    landmark['x'] * img.shape[1],
                ), int(landmark['y'] * img.shape[0])
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Vẽ điểm landmark

        # Lưu ảnh kết quả
        output_path = 'output_pose_landmarks.png'
        cv2.imwrite(output_path, img)
        self.assertTrue(
            cv2.imread(output_path) is not None,
            'Không thể lưu ảnh kết quả',
        )


if __name__ == '__main__':
    unittest.main()
