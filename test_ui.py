"""Regression checks for asynchronous analysis and stale-result handling."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from PyQt5.QtCore import QCoreApplication, QEvent, QThread, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from src.ui.view.main_screen import MainScreen

RESULT = {'class_id': -1, 'top_class': 2, 'score': .6, 'scores': [.1,.1,.6,.1,.1]}

class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.image = Path(self.temp.name) / 'retina.png'
        pix = QImage(100, 100, QImage.Format_RGB32)
        pix.fill(Qt.red)
        pix.save(str(self.image))
        model = SimpleNamespace(threshold=.8, display_name='Test', predict=lambda path: RESULT)
        self.window = MainScreen(model=model)
        self.window.show()

    def tearDown(self):
        self.window.busy = False
        self.window.close()
        self.temp.cleanup()

    def wait_idle(self):
        deadline = time.monotonic() + 10
        while self.window.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertFalse(self.window.busy)
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_escape_does_not_close_busy_window(self):
        self.window.busy = True
        QTest.keyClick(self.window, Qt.Key_Escape)
        self.assertTrue(self.window.isVisible())

    def test_completed_workers_are_released(self):
        self.window.set_image(self.image)
        for _ in range(2):
            self.window.analyze()
            self.wait_idle()
            self.assertIsNone(self.window.worker)
            self.assertEqual(self.window.findChildren(QThread), [])
        self.assertEqual(self.window.last_result, RESULT)

    def test_new_image_and_clear_remove_old_result(self):
        self.window.set_image(self.image)
        self.window.show_result(RESULT)
        self.assertTrue(self.window.export_button.isEnabled())
        self.window.set_image(self.image)
        self.assertIsNone(self.window.last_result)
        self.assertFalse(self.window.export_button.isEnabled())
        self.window.clear_image()
        self.assertIsNone(self.window.image_path)
        self.assertFalse(self.window.analyze_button.isEnabled())

    def test_bad_model_preserves_active_model(self):
        old_model = self.window.model
        self.window.set_image(self.image)
        self.window.show_result(RESULT)
        self.window.load_model(Path(self.temp.name) / 'missing.pt')
        self.wait_idle()
        self.assertIs(self.window.model, old_model)
        self.assertIsNone(self.window.last_result)
        self.assertTrue(self.window.analyze_button.isEnabled())

if __name__ == '__main__':
    unittest.main()
