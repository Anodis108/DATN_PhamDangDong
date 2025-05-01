from __future__ import annotations

import base64

import cv2
import numpy as np


def image_to_base64(image: np.ndarray, format: str = '.png') -> str:
    """
    Chuyển ảnh NumPy array thành chuỗi base64.
    :param image: Ảnh đầu vào (ndarray dạng RGB hoặc BGR)
    :param format: Định dạng ảnh (mặc định PNG)
    :return: Chuỗi base64 đại diện ảnh
    """
    success, encoded_image = cv2.imencode(format, image)
    if not success:
        raise ValueError('Encoding ảnh thất bại.')
    base64_bytes = base64.b64encode(encoded_image.tobytes())
    return base64_bytes.decode('utf-8')


def base64_to_image(base64_string: str) -> np.ndarray:
    """
    Chuyển chuỗi base64 thành ảnh NumPy array.
    :param base64_string: Chuỗi base64 ảnh
    :return: Ảnh dạng NumPy array
    """
    image_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # ảnh BGR
    return image
