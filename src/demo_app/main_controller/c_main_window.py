from __future__ import annotations

import logging
import os
from datetime import datetime
from queue import Queue

import numpy as np
from common.utils import get_settings
from PIL import Image
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QEvent
from PyQt5.QtCore import QObject
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel
from thread import ThreadWork
from uis.main_window import Ui_MainWindow

from .utils import ImageUtils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
settings = get_settings()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('Detect Hat, Helmet and Mask')

        self.camera_path = settings.app.camera_path
        self.image_path = None
        self.is_video = False
        self.capture_queue = Queue()
        self.stream_queue = Queue()
        self.timer = QtCore.QTimer(self)

        self.ui.le_cam_path.setText(self.camera_path)
        self.ui.label_avatar.setPixmap(QPixmap(settings.app.img_logo_path))
        self.connect_signals()

        # Khởi tạo thread_work
        self.thread_work = None

    def connect_signals(self):
        self.ui.btn_choose_path.clicked.connect(self.choose_path_img)
        self.ui.btn_start.clicked.connect(self.start_processing)
        self.ui.btn_link_cam.clicked.connect(self.link_cam)
        self.ui.btn_clear.clicked.connect(
            lambda: self.update_results(mode='clear'),
        )
        self.ui.btn_pause.clicked.connect(self.pause)

    def link_cam(self):
        self.ui.le_cam_path.setText('0')

    def choose_path_img(self):
        options = QtWidgets.QFileDialog.Options()
        base_dir = os.getcwd()
        default_dir = os.path.join(base_dir, 'resource/images')
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            'Choose File',
            default_dir,
            'All Supported Files (*.png *.jpg *.jpeg *.bmp *.gif *.mp4 *.avi *.mkv);;'
            'Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;'
            'Video Files (*.mp4 *.avi *.mkv);;All Files (*)',
            options=options,
        )
        if file_path:
            self.image_path = file_path
            self.is_video = os.path.splitext(self.image_path)[1].lower() in [
                '.mp4', '.avi', '.mkv',
            ]
            self.ui.le_cam_path.setText(file_path)
        else:
            self.image_path = None

    def start_processing(self):
        if not hasattr(self, 'image_path') or not self.image_path:
            self.image_path = self.ui.le_cam_path.text().strip()
        if not isinstance(self.image_path, str) or not self.image_path:
            logging.error(f'No image path provided: {self.image_path}')
            QtWidgets.QMessageBox.warning(
                self, 'Error', 'Please select an image or enter a valid path.',
            )
            return
        if self.image_path != '0' and not os.path.exists(self.image_path):
            logging.error(f'Image file does not exist: {self.image_path}')
            QtWidgets.QMessageBox.warning(
                self, 'Error', f'Image file not found: {self.image_path}',
            )
            return

        self.capture_queue = Queue()
        self.stream_queue = Queue()

        # Tạo và chạy ThreadWork
        self.thread_work = ThreadWork(self.image_path, self.is_video)
        self.thread_work.image_processed.connect(self.on_image_processed)
        self.thread_work.video_frame_processed.connect(
            self.on_video_frame_processed,
        )
        self.thread_work.error_occurred.connect(self.on_error_occurred)
        self.thread_work.finished.connect(self.on_thread_finished)
        self.thread_work.start()

        # Nếu là video, bắt đầu timer để xử lý khung hình
        if self.is_video or self.image_path == '0':
            self.timer.timeout.connect(self.thread_work.process_video_frame)
            self.timer.start(30)

    def on_image_processed(self, image_rgb, place, response, save_img_path):
        # Hiển thị ảnh lên UI
        self.display_image_on_label(self.ui.label_input, image_rgb)
        self.display_results(image_rgb, place, response, save_img_path)

    def on_video_frame_processed(self, rgb_frame):
        # Hiển thị khung hình video
        self.display_image_on_label(self.ui.label_input, rgb_frame)

    def on_error_occurred(self, error_message):
        QtWidgets.QMessageBox.warning(self, 'Error', error_message)

    def on_thread_finished(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.thread_work:
            self.thread_work.stop()
            self.thread_work = None

    def pause(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(30)

    def display_results(self, image: np.ndarray, place: str, height: float, save_img_path: str):
        self.update_results(mode='push')
        self.display_image_on_label(self.ui.label_img_result1, image)
        self.ui.label_img_result1.setProperty('img_path', save_img_path)
        self.ui.label_time_result1.setText(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )
        self.ui.label_place_result1.setText(place)
        self.ui.label_height_result1.setText(str(height))

    def update_results(self, mode: str = 'push'):
        label_img_results = [
            self.ui.label_img_result1, self.ui.label_img_result2,
            self.ui.label_img_result3, self.ui.label_img_result4,
        ]
        label_time_results = [
            self.ui.label_time_result1, self.ui.label_time_result2,
            self.ui.label_time_result3, self.ui.label_time_result4,
        ]
        label_place_results = [
            self.ui.label_place_result1, self.ui.label_place_result2,
            self.ui.label_place_result3, self.ui.label_place_result4,
        ]
        label_height_results = [
            self.ui.label_height_result1, self.ui.label_height_result2,
            self.ui.label_height_result3, self.ui.label_height_result4,
        ]

        if mode == 'push':
            for i in range(3, 0, -1):
                label_img_results[i].clear()
                if label_img_results[i - 1].pixmap() is not None:
                    label_img_results[i].setPixmap(
                        label_img_results[i - 1].pixmap().copy(),
                    )

                # 🔧 Copy img_path của label trước đó
                prev_path = label_img_results[i - 1].property('img_path')
                if prev_path:
                    label_img_results[i].setProperty('img_path', prev_path)

                label_time_results[i].setText(label_time_results[i - 1].text())
                label_place_results[i].setText(
                    label_place_results[i - 1].text(),
                )
                label_height_results[i].setText(
                    label_height_results[i - 1].text(),
                )
        elif mode == 'clear':
            for i in range(4):
                label_img_results[i].setPixmap(
                    QPixmap('./resources/icons/folder_icon.png'),
                )
                label_time_results[i].setText('Time')
                label_place_results[i].setText('')
                label_height_results[i].setText('')
        for label in label_img_results:
            label.setAlignment(Qt.AlignCenter)
            label.installEventFilter(self)

    def display_image_on_label(self, ui_label: QLabel, image: Image.Image):
        """Hiển thị ảnh PIL.Image lên QLabel sau khi convert sang RGB."""
        # Đảm bảo ảnh ở định dạng RGB
        image = image.convert('RGB')

        # Convert PIL → NumPy
        image_np = np.array(image)

        # Tạo QImage từ NumPy
        height, width, channel = image_np.shape
        bytes_per_line = 3 * width
        qimage = QtGui.QImage(
            image_np.data, width, height,
            bytes_per_line, QtGui.QImage.Format_RGB888,
        )

        # Hiển thị lên QLabel
        ui_label.clear()
        ui_label.setScaledContents(True)
        ui_label.setPixmap(QtGui.QPixmap.fromImage(qimage))

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if isinstance(obj, QLabel) and event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            path = obj.property('img_path')
            if path:
                self.img_utils = ImageUtils()  # Khởi tạo ImageUtils nếu cần
                self.img_utils.show_image_window(path)
            return True
        return super().eventFilter(obj, event)
