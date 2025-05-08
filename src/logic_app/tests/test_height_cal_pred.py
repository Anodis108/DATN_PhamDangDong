from __future__ import annotations

import json
import unittest

import cv2
from app.height_cal_pred import HeightInput
from app.height_cal_pred import HeightService
from common.utils import get_settings
from fastapi.encoders import jsonable_encoder


class TestHeightService(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = get_settings()
        self.height_service = HeightService(settings=self.settings)
        self.img_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_PhamDangDong/DATN_PhamDangDong/resource/data/data_test/z6194149941298_733dfe9d1f76e5f775c0f9336c4b3f3e.jpg'

    def test_process_height_service(self):
        test_image = cv2.imread(self.img_path)
        inputs = HeightInput(image=test_image)
        result = self.height_service.process(inputs=inputs)

        # Chuyển đổi kết quả thành JSON hợp lệ và in ra
        json_result = jsonable_encoder(result)
        print(json.dumps(json_result, indent=4))


if __name__ == '__main__':
    unittest.main()
