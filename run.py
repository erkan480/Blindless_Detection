import sys
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QApplication

from src.ui.view.get_started_screen import GetStartedScreen

if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    get_started = GetStartedScreen()
    screen = app.primaryScreen().availableGeometry()
    get_started.resize(min(1100, screen.width()-40), min(780, screen.height()-60))
    sys.exit(app.exec_())
