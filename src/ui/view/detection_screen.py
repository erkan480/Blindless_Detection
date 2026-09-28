import random

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QLabel, QDialog

from src.ai.load_model import LoadModel
from src.module.ui_module import icon_folder

# gerekli modüller import ediliyor.


class DetectionScreen(QDialog):
    def __init__(self,path,model):
        super().__init__()
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.detected_disease = self.get_random_disease(path,model)

        self.init_ui()


    def get_random_disease(self,path,model):
        disease_id = LoadModel(model).get_disease(path)
        if disease_id not in range(0,5):
            disease_id = 5

        file_name = f"disease_{disease_id}"
        image = f"{icon_folder}/{file_name}.jpg"

        return image


    def init_ui(self):
        x,y = 1600,900

        image = QLabel(self)
        image.setPixmap(QPixmap(self.detected_disease))
        image.adjustSize()

        self.setFixedSize(x,y)
        self.setWindowTitle("HASTALIK TEŞHİSİ")
        self.setWindowIcon(QIcon(icon_folder + "glasses.png"))