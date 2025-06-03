from __future__ import annotations

import json  # Thêm để định dạng JSON
import os
import unittest

import requests  # type: ignore


class TestHeightPredAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.api_url = 'http://localhost:5000/v1/height_pred'
        self.distance = [
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0
            ]
        ]

    def test_height_pred_api(self):
        payload = {
            'x': self.distance
        }
        response = requests.post(self.api_url, json=payload)

        print('Status Code:', response.status_code)

        # In JSON ra terminal theo định dạng đẹp
        print('Response JSON:')
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))

        # Các kiểm tra
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn('message', json_data)
        self.assertIn('info', json_data)
        self.assertIn('bboxes', json_data['info'])
        self.assertIsInstance(json_data['info']['bboxes'], list)


if __name__ == '__main__':
    unittest.main()
