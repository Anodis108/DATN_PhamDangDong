from __future__ import annotations

import os
import signal
import sys

from main_controller.c_main_window import MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtCore import QLibraryInfo
print(QLibraryInfo.location(QLibraryInfo.PluginsPath))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
