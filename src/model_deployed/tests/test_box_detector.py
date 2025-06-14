from __future__ import annotations

import json
import unittest

import cv2
from common import get_settings
from infrastructure.box_detector import BoxDetectorModel
from infrastructure.box_detector import BoxDetectorModelInput


class TestBoxDetector(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = get_settings()
        self.box_detector_model = BoxDetectorModel(settings=self.settings)

    def test_box_detect(self):
        image_path = '/mnt/d/project/DATN/DATN_PhamDangDong/resource/data/data_test/z6194149941298_733dfe9d1f76e5f775c0f9336c4b3f3e.jpg'
        img = cv2.imread(image_path)

        # Kiểm tra giá trị pixel đầu tiên (tại vị trí 0,0)
        b, g, r = img[0, 0]
        if b != r:
            print('Ảnh đang ở định dạng BGR')
        else:
            print('Ảnh có thể là RGB hoặc ảnh đơn sắc')

        inputs = BoxDetectorModelInput(img=img)

        import asyncio
        outputs = asyncio.run(self.box_detector_model.process(inputs))

        # Chuẩn bị dữ liệu đầu ra dưới dạng dictionary
        result = {
            'bboxes': outputs.bboxes.tolist() if hasattr(outputs.bboxes, 'tolist') else outputs.bboxes.tolist(),
            'scores': outputs.scores.tolist() if hasattr(outputs.scores, 'tolist') else outputs.scores.tolist(),
            'pixel_per_cm': outputs.pixel_per_cm,
        }

        # In ra kết quả dưới dạng JSON
        print(json.dumps(result, indent=4))

        # Kiểm tra nếu có bounding box và vẽ nó lên ảnh
        if len(outputs.bboxes) > 0:
            bbox = outputs.bboxes[0][:4]  # Chỉ truy cập nếu có bounding box
            confidence = outputs.scores[0]
            print(bbox, confidence)
        else:
            print('Không phát hiện bounding box nào.')

        # Vẽ bounding box lên ảnh
        bbox = outputs.bboxes[0][:4]
        confidence = outputs.scores[0]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, f'Conf: {confidence:.2f}', (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )

        cv2.imwrite('output.png', img)


if __name__ == '__main__':
    unittest.main()
