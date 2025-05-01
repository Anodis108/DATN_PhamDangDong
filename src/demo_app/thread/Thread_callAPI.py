from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import requests  # type: ignore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import QThread


class APICallerThread(QThread):
    def __init__(self):
        super().__init__()

    def run(self):
        self.exec_()

    def prepare_image_file(self, image: np.ndarray):
        """Convert np.ndarray image to file-like object for upload"""
        _, buffer = cv2.imencode('.jpg', image)
        file_bytes = BytesIO(buffer.tobytes())
        file_bytes.name = 'image.jpg'
        return file_bytes

    @pyqtSlot(str, object)
    def call_api(self, api_url, data):
        try:
            if isinstance(data, np.ndarray):
                # Nếu là ảnh numpy
                file_bytes = self.prepare_image_file(data)
                files = {'file': (file_bytes.name, file_bytes, 'image/jpeg')}
                response = requests.post(api_url, files=files)
            else:
                # Gửi JSON thông thường
                response = requests.post(api_url, json=data)

            response.raise_for_status()
            print(f'Phản hồi từ {api_url}: {response.json()}')

        except requests.RequestException as e:
            print(f'Lỗi khi gọi {api_url}: {e}')
