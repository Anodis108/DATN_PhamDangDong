from __future__ import annotations

import os
from os.path import dirname

ROOT = dirname(__file__)

# video_path = r"resource/data/real_test.mp4"
camera_path = r'rtsp://root:Atin@123@192.168.1.235/axis-media/media.amp'
model_path_detect = os.path.join(
    ROOT, r'resource/weights/last_detect_head_07042025_v2_y8m_640.pt',
)
# print("Model path detection", model_path_detect)
model_path_segment = os.path.join(
    ROOT,  r'resource/weights/last_seg_07042025.pt',
)
audio_path = os.path.join(ROOT,  r'resource/audio')
img_result_path = os.path.join(ROOT,  r'resource/img')

img_logo_path = os.path.join(ROOT, r'logo.ico')
SERVER_IP = '192.168.1.78'  # IP của Raspberry Pi
PORT = 12345
TIME_TO_PUSH_EVENT = 5
