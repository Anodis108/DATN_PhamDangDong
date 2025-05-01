from __future__ import annotations

import unittest

import cv2
from common import get_settings
from infrastructure.box_detector import CardDetectorModel
from infrastructure.box_detector import CardDetectorModelInput


class TestCardDetector(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = get_settings()
        self.card_detector_model = CardDetectorModel(settings=self.settings)

    def test_detect(self):
        image_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/resource/data/demo_data_card/LuThihSen3.jpg'
        img = cv2.imread(image_path)
        print(img)
        # Chuyển sang RGB
        # Kiểm tra giá trị pixel đầu tiên (tại vị trí 0,0)
        b, g, r = img[0, 0]
        # Nếu giá trị B khác giá trị R, nhiều khả năng ảnh đang ở định dạng BGR
        if b != r:
            print('Ảnh đang ở định dạng BGR')
        else:
            print('Ảnh có thể là RGB hoặc ảnh đơn sắc')
        inputs = CardDetectorModelInput(
            img=img,
        )
        import asyncio
        outputs = asyncio.run(self.card_detector_model.process(inputs))
        print(outputs.bboxes)
        print('-------')
        if len(outputs.bboxes) > 0:
            bbox = outputs.bboxes[0][:4]  # Chỉ truy cập nếu có bounding box
            confidence = outputs.scores[0]
            print(bbox, confidence)
        else:
            print('Không phát hiện bounding box nào.')

        # print(outputs.kpss)
        print(outputs)
        bbox = outputs.bboxes[0][:4]
        confidence = outputs.bboxes[0][-1]
        # Vẽ bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, f'Conf: {confidence:.2f}', (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )

        cv2.imwrite('output.png', img)


if __name__ == '__main__':
    unittest.main()
