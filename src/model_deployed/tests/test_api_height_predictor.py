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
                9.933471372616063,57.19990722740044,67.03142189601489,87.5611880572484,32.26543209272624,5.833722614191923
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



if __name__ == '__main__':
    unittest.main()
