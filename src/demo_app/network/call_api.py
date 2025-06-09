from __future__ import annotations

from io import BytesIO
from typing import Union

import cv2
import numpy as np
import requests  # type: ignore
from common.bases import BaseModel
from model import BaseResults


class APICaller(BaseModel):

    @staticmethod
    def prepare_image_file(image: np.ndarray, img_path: str):
        """Convert np.ndarray image to file-like object for upload"""
        _, buffer = cv2.imencode('.jpg', image)
        file_bytes = BytesIO(buffer.tobytes())
        file_bytes.name = img_path
        return file_bytes

    @staticmethod
    def call_api(api_url: str, data: Union[np.ndarray, BaseModel], img_path: str) -> dict | None:
        try:
            if isinstance(data, np.ndarray):
                file_bytes = APICaller.prepare_image_file(data, img_path)
                files = {'file': (file_bytes.name, file_bytes, 'image/jpeg')}
                response = requests.post(api_url, files=files)

            elif isinstance(data, BaseModel):
                response = requests.post(api_url, json=data.dict())

            else:
                raise ValueError('Unsupported data type for API call')

            response.raise_for_status()
            print('DEBUG response:', response.json())
            return BaseResults(
                heights=response.json()['info']['results'],
                out_path=response.json()['info']['out_path'],
            )

        except requests.RequestException as e:
            print(f'Lỗi khi gọi {api_url}: {e}')
            return None  # ✅ Trả về None khi có lỗi
