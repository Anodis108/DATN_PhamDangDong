# from __future__ import annotations
# import unittest
# import numpy as np
# import requests  # type: ignore
# class TestFaceEmbedding(unittest.TestCase):
#     def setUp(self) -> None:
#         pass
#     def test_api_embedding(self):
#         loaded_embedding = np.load(
#             '/home/nguyen.luong.hung@sun-asterisk.com/PoC/AI_Face-login/src/model_deployed/tests/embedding.npy',
#         )
#         print(
#             'Image shape:', loaded_embedding.shape,
#             'dtype:', loaded_embedding.dtype,
#         )
#         kpoint = np.array([
#             [191.55109, 180.67368],
#             [234.27043, 179.21089],
#             [214.12573, 203.67413],
#             [198.07564, 226.53897],
#             [230.68082, 225.25124],
#         ])
#         print('kpoint shape:', kpoint.shape, 'dtype:', kpoint.dtype)
#         payload = {
#             'image': loaded_embedding.tolist(),
#             'landmarks': kpoint.tolist(),
#         }
#         result = requests.post(
#             'http://localhost:5000/v1/embedding', json=payload,
#         )
#         print(result.json())
from __future__ import annotations
