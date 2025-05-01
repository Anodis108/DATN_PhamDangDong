from __future__ import annotations

import asyncio
import unittest

import cv2
from common import get_settings
from infrastructure.pose_detector import TextDetectorModel
from infrastructure.pose_detector import TextDetectorModelInput


class TestTextDetectorOnly(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = get_settings()
        self.text_detector = TextDetectorModel(settings=self.settings)

    def test_text_detector(self):
        # Đường dẫn ảnh cần test
        image_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/src/model_deployed/processed_output.jpg'
        img = cv2.imread(image_path)
        self.assertIsNotNone(img, 'Ảnh không được load thành công!')

        # Dữ liệu bbox từ CardDetectorModel
        # bbox = [x1, y1, x2, y2]

        # Tạo input
        text_input = TextDetectorModelInput(
            img_processed=img,
        )

        # Gọi hàm xử lý
        text_output = asyncio.run(self.text_detector.process(text_input))

        # In kết quả
        print('Classes:', text_output.class_list)
        print('Bboxes:', text_output.bboxes_list)
        print('Confidences:', text_output.conf_list)

        # # Vẽ kết quả lên ảnh
        # img_output = text_output.processed_images.copy()
        # for cls, conf, box in zip(
        #     text_output.class_list,
        #     text_output.conf_list,
        #     text_output.bboxes_list,
        # ):
        #     x1, y1, x2, y2 = map(int, box)
        #     cv2.rectangle(img_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(
        #         img_output,
        #         f'{cls}:{conf:.2f}',
        #         (x1, y1 - 10),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (0, 255, 0),
        #         2,
        #     )

        # Lưu ảnh
        # cv2.imwrite('text_detector_output.png', img_output)
        cv2.imwrite(
            'text_detector_preproces.png',
            img,
        )


if __name__ == '__main__':
    unittest.main()
