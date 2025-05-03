from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from queue import Queue

import cv2
import numpy as np
from common.settings import Settings
from config import camera_path
from config import img_logo_path
from config import save_dir
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QEvent
from PyQt5.QtCore import QObject
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QVBoxLayout
from thread.Thread_callAPI import APICallerThread
from uis.main_window import Ui_MainWindow
# import folium

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('Detect Hat, Helmet and Mask')

        self.create_queue()
        self.start_time = 0
        self.connect_signal()
        self.timer = QtCore.QTimer(self)
        self.place = 'Detect Hat, \nHelmet and Mask'
        self.level = ''
        # self.type = ""
        self.obj_center = ()
        self.current_obj_center = ()
        self.color = None

        self.camera_path = camera_path
        self.ui.le_cam_path.setText(self.camera_path)

        pixmap = QPixmap(img_logo_path)
        self.ui.label_avatar.setPixmap(pixmap)

        self.is_video = False
        self.setting = Settings()
        self.image_path = None
        self.test = 'test'

        # self.ui.frame_2.setVisible(False)

    def connect_signal(self):
        self.ui.btn_choose_path.clicked.connect(self.choose_path_img)
        self.ui.btn_start.clicked.connect(self.start_thread)
        self.ui.btn_link_cam.clicked.connect(self.link_cam)
        self.ui.btn_clear.clicked.connect(
            lambda: self.update_results(mode='clear'),
        )
        self.ui.btn_pause.clicked.connect(self.pause)

    def link_cam(self):
        self.ui.le_cam_path.setText(0)

    def choose_path_img(self):
        options = QtWidgets.QFileDialog.Options()
        base_dir = os.getcwd()
        default_dir = os.path.join(base_dir, r'resource/images')
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            'Choose File',
            default_dir,
            'All Supported Files (*.png *.jpg *.jpeg *.bmp *.gif *.mp4 *.avi *.mkv);;Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;Video Files (*.mp4 *.avi *.mkv);;All Files (*)',
            options=options,
        )

        if file_path:
            self.image_path = file_path
            # Kiểm tra phần mở rộng
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            self.is_video = ext in ['.mp4', '.avi', '.mkv']
            self.ui.le_cam_path.setText(file_path)
        else:
            self.image_path = None

    def create_queue(self):
        self.capture_queue = Queue()
        self.stream_queue = Queue()

    def start_thread(self):

        self.create_queue()
        self.create_threads()

    def run_image_processing(self):
        image = cv2.imread(self.image_path)
        if image is not None:
            self.display_results(
                image, self.test, self.test,
                self.test, self.test, self.test, self.test,
            )
            self.display_image_on_label(self.ui.label_input, image)
        else:
            logging.warning('Không đọc được ảnh.')
            return

        # Kiểm tra thread_callapi có tồn tại không
        if not hasattr(self, 'thread_callapi') or self.thread_callapi is None:
            self.thread_callapi = APICallerThread()
            self.thread_callapi.start()

        self.thread_callapi.call_api(self.setting.host_ocr_service, image)

    def run_video_stream(self):
        self.video_capture = cv2.VideoCapture(self.image_path)
        self.timer.timeout.connect(self.read_video_frame)
        self.timer.start(30)  # 30 ms ~ 33 fps

    def pause(self):
        pass

    def create_threads(self):
        self.thread_callapi = APICallerThread()
        self.thread_callapi.start()

        if not hasattr(self, 'image_path') or not self.image_path:
            return

        if self.is_video:
            # Nếu là video → chạy xử lý luồng
            self.run_video_stream()
        else:
            # Nếu là ảnh → hiển thị ảnh
            self.run_image_processing()

    def display_results(self, fire_image, place, level, nhan):
        self.update_results(mode='push')
        save_img_path, timestamp = self.save_img(fire_image, save_dir)

        self.display_image_on_label(self.ui.label_img_result1, fire_image)

        # Gán đường dẫn ảnh vào property "img_path"
        self.ui.label_img_result1.setProperty('img_path', save_img_path)

        self.ui.label_time_result1.setText(
            str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        )
        self.ui.label_place_result1.setText(place)
        self.ui.label_level_result1.setText(level)
        self.ui.lb_nhan_1.setText(nhan)

    def update_results(self, mode='push'):
        label_img_results = [
            self.ui.label_img_result1, self.ui.label_img_result2, self.ui.label_img_result3,
            self.ui.label_img_result4,
        ]

        label_time_results = [
            self.ui.label_time_result1, self.ui.label_time_result2, self.ui.label_time_result3,
            self.ui.label_time_result4,
        ]

        label_place_results = [
            self.ui.label_place_result1, self.ui.label_place_result2, self.ui.label_place_result3,
            self.ui.label_place_result4,
        ]
        label_level_results = [
            self.ui.label_level_result1, self.ui.label_level_result2, self.ui.label_level_result3,
            self.ui.label_level_result4,
        ]
        lb_nhan_ = [
            self.ui.lb_nhan_1, self.ui.lb_nhan_2,
            self.ui.lb_nhan_3, self.ui.lb_nhan_4,
        ]

        if mode == 'push':
            for i in range(3, 0, -1):
                label_img_results[i].clear()
                label_img_results[i].setPixmap(label_img_results[i-1].pixmap())
                old_path = label_img_results[i-1].property('img_path')
                label_img_results[i].setProperty('img_path', old_path)
                label_time_results[i].setText(label_time_results[i-1].text())
                label_place_results[i].setText(label_place_results[i-1].text())
                label_level_results[i].setText(label_level_results[i-1].text())
                lb_nhan_[i].setText(lb_nhan_[i-1].text())
        elif mode == 'clear':
            for i in range(4):
                label_img_results[i].setPixmap(
                    QtGui.QPixmap('./resources/icons/folder_icon.png'),
                )
                label_time_results[i].setText('Time')
                label_place_results[i].setText('')
                label_level_results[i].setText('')
                lb_nhan_[i].setText('')
        for label in label_img_results:
            label.setAlignment(Qt.AlignCenter)
            label.installEventFilter(self)

    def display_image_on_label(self, ui_label, image):
        pixmap = QtGui.QImage(
            image.data, image.shape[1], image.shape[0], QtGui.QImage.Format.Format_RGB888,
        ).rgbSwapped()
        ui_label.clear()
        ui_label.setPixmap(QtGui.QPixmap.fromImage(pixmap))

    def paintEvent(self, e):
        # self.obj_center = (0, 0)
        # if self.stream_queue.qsize() > 0:
        #     text, origin_image = self.stream_queue.get()
        #     image = origin_image.copy()
        #     warning_level = self.thread_work.event.warning_level
        #     if warning_level != "200":
        #         # print(warning_level)
        #         current_time = time.time()
        #         if (current_time - self.start_time) >= TIME_TO_PUSH_EVENT:
        #             self.display_results(image, self.place, text)
        #             self.save_img(image, img_result_path)
        #             # self.thread_client.send_message(warning_level)
        #             self.start_time = current_time
        #             self.thread_audio.play_audio(warning_level)
        #             # Hiệu ứng cảnh báo đỏ
        #             red_image = np.zeros(image.shape, image.dtype)
        #             red_image[:, :] = (0, 0, 255)
        #             red_factor = 0.2
        #             image = cv2.addWeighted(red_image, red_factor, image, 1 - red_factor, 0)
        #     self.display_image_on_label(self.ui.label_input,image)

        self.update()

    def read_video_frame(self):
        if self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_image_on_label(self.ui.label_input, rgb_frame)
            else:
                self.timer.stop()
                self.video_capture.release()
                logging.info('Đã đọc hết video.')
        else:
            self.timer.stop()

    def save_img(self, image: np.ndarray, save_dir: str):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'image_{timestamp}.jpg'
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, image)
        return save_path, timestamp

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if isinstance(obj, QLabel) and event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            path = obj.property('img_path')
            if path:
                self.show_image_window(path)
            return True
        return super().eventFilter(obj, event)

    def show_image_window(self, img_path):
        window = QDialog()
        window.setWindowTitle('Hiển thị ảnh')
        window.resize(1440, 810)
        layout = QVBoxLayout()
        window.setLayout(layout)
        label = QLabel()
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            label.setText('Không thể đọc ảnh.')
        else:
            label.setPixmap(pixmap.scaled(1440, 810, Qt.KeepAspectRatio))
            label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        window.exec_()

    def send_data_to_api(self, api_url, data):
        self.api_thread.call_api(api_url, data)
