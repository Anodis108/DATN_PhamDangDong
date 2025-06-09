from __future__ import annotations

import os
import signal
import sys

from main_controller.c_main_window import MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtCore import QLibraryInfo

# 👉 Thiết lập path chính xác đến thư mục plugin Qt
qt_plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path

print(f"[DEBUG] QT Plugin path: {qt_plugin_path}")

# 👉 Cho phép override lỗi thư viện trùng nhau (thường do OpenMP)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
