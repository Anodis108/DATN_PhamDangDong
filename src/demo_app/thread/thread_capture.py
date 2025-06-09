from __future__ import annotations

import logging
import time
from queue import Full
from queue import Queue

import cv2
from PyQt5.QtCore import QThread

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


class ThreadCapture(QThread):
    def __init__(self, video_path, capture_queue):
        super().__init__()

        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.running = False
        self.capture_queue: Queue = capture_queue
        self.max_buffer_size = 50
        self.pause_flag = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # logging.warning("[ThreadCapture] Video hết, quay lại từ đầu.")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            try:
                if self.capture_queue.qsize() < self.max_buffer_size:
                    self.capture_queue.put(frame)
                else:
                    continue
                    # logging.warning("[ThreadCapture] Queue đầy, bỏ frame để tránh lag.")
            except Full:
                pass

            time.sleep(0.001)
        self.cap.release()

    def stop(self):
        self.running = False
        logging.info('[ThreadCapture] Dừng luồng ghi hình.')
        self.quit()
        self.wait()
