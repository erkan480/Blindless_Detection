"""Model selection and retinal image analysis, with all inference off the GUI thread."""
import json
from datetime import datetime
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QComboBox, QFrame, QWidget, QScrollArea, QSizePolicy, QMessageBox)
from src.ui.view.retina_style import STYLE

ROOT = Path(__file__).resolve().parents[3]
LABELS = ['Diyabetik retinopati bulgusu yok', 'Hafif', 'Orta', 'Ağır', 'İleri (proliferatif)']
FORMATS = {'.png', '.jpg', '.jpeg', '.bmp'}
AUTHOR = 'Erkan Çolak'

def label(text, role=None):
    widget = QLabel(text)
    widget.setTextFormat(Qt.PlainText)
    if role: widget.setObjectName(role)
    return widget

def card():
    widget = QFrame()
    widget.setObjectName('card')
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(20, 18, 20, 18)
    layout.setSpacing(12)
    return widget, layout

def percentage(value):
    return f'%{value * 100:.1f}'.replace('.', ',')

class ModelThread(QThread):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    def __init__(self, path, do_import=False, parent=None):
        super().__init__(parent)
        self.path, self.do_import = path, do_import
    def run(self):
        try:
            from src.ai.retina import RetinaPredictor, import_model
            path = import_model(self.path) if self.do_import else self.path
            self.ready.emit(RetinaPredictor(path))
        except Exception as exc:
            self.failed.emit(str(exc))

class PredictThread(QThread):
    result = pyqtSignal(object)
    failed = pyqtSignal(str)
    def __init__(self, model, path, parent=None):
        super().__init__(parent)
        self.model, self.path = model, path
    def run(self):
        try: self.result.emit(self.model.predict(self.path))
        except Exception as exc: self.failed.emit(str(exc))

class MainScreen(QDialog):
    def __init__(self, model=None, auto_load=False):
        super().__init__()
        self.model, self.worker, self.model_worker = model, None, None
        self.image_path, self.last_result = None, None
        self.original_pixmap = None
        self.busy = False
        self.setObjectName('root')
        self.setStyleSheet(STYLE)
        self.setWindowTitle(f'Blindless Detection • Retina analizi • {AUTHOR}')
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(1100, 780)
        self.setMinimumSize(880, 640)
        self.setAcceptDrops(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        page = QVBoxLayout(content)
        page.setContentsMargins(22, 18, 22, 16)
        page.setSpacing(14)

        header = QHBoxLayout()
        mark = label('BD', 'brand')
        mark.setFixedSize(48, 48)
        mark.setAlignment(Qt.AlignCenter)
        header.addWidget(mark)
        titles = QVBoxLayout()
        titles.setSpacing(1)
        titles.addWidget(label('Retina görüntüsü analizi', 'title'))
        titles.addWidget(label('Diyabetik retinopati evreleri için görüntü değerlendirmesi.', 'muted'))
        header.addLayout(titles)
        header.addStretch()
        self.badge = label('Model seçin', 'badge')
        header.addWidget(self.badge)
        page.addLayout(header)

        model_card, ml = card()
        model_row = QHBoxLayout()
        model_text = QVBoxLayout()
        model_text.addWidget(label('Analiz modeli', 'section'))
        self.model_info = label('Eğitilmiş bir model seçin veya Colab çıktısını ekleyin.', 'muted')
        self.model_info.setWordWrap(True)
        model_text.addWidget(self.model_info)
        model_row.addLayout(model_text, 1)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(235)
        self.model_combo.setAccessibleName('Eğitilmiş model seçimi')
        self.model_combo.activated.connect(self.select_model)
        model_row.addWidget(self.model_combo)
        self.import_button = QPushButton('Model ekle…')
        self.import_button.setToolTip('Colab eğitim defterinin ürettiği uyumlu .pt modelini ekleyin.')
        self.import_button.clicked.connect(self.import_model)
        model_row.addWidget(self.import_button)
        ml.addLayout(model_row)
        page.addWidget(model_card)

        body = QHBoxLayout()
        body.setSpacing(16)
        image_card, il = card()
        il.addWidget(label('1  /  Görüntüyü hazırlayın', 'section'))
        self.preview = label('Retina görüntüsünü buraya sürükleyin\nveya aşağıdaki düğmeyle seçin.', 'preview')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(340, 275)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        il.addWidget(self.preview, 1)
        self.file_info = label('PNG, JPG veya BMP • Görüntüler bu bilgisayarda işlenir.', 'muted')
        self.file_info.setWordWrap(True)
        il.addWidget(self.file_info)
        image_actions = QHBoxLayout()
        self.button = QPushButton('Görüntü seç…')
        self.button.clicked.connect(self.open_file)
        self.clear_button = QPushButton('Temizle')
        self.clear_button.clicked.connect(self.clear_image)
        image_actions.addWidget(self.button, 1)
        image_actions.addWidget(self.clear_button)
        il.addLayout(image_actions)
        self.analyze_button = QPushButton('Analizi başlat')
        self.analyze_button.setObjectName('primary')
        self.analyze_button.clicked.connect(self.analyze)
        il.addWidget(self.analyze_button)
        body.addWidget(image_card, 5)

        result_card, rl = card()
        rl.setSpacing(8)
        result_card.setMinimumWidth(345)
        rl.addWidget(label('2  /  Model değerlendirmesi', 'section'))
        self.result_tag = label('ANALİZ BEKLENİYOR', 'eyebrow')
        rl.addWidget(self.result_tag)
        self.result_title = label('Henüz sonuç yok', 'resultTitle')
        self.result_title.setWordWrap(True)
        rl.addWidget(self.result_title)
        self.status = label('Bir görüntü seçip analizi başlatın.', 'muted')
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(48)
        rl.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        rl.addWidget(self.progress)
        rl.addSpacing(4)
        rl.addWidget(label('Sınıflara göre model skoru', 'section'))
        self.bars, self.score_labels = [], []
        for i, name in enumerate(LABELS):
            row = QHBoxLayout()
            row.addWidget(label(name))
            row.addStretch()
            score = label('—', 'muted')
            score.setMinimumWidth(50)
            score.setAlignment(Qt.AlignRight)
            row.addWidget(score)
            rl.addLayout(row)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setValue(0)
            rl.addWidget(bar)
            self.bars.append(bar)
            self.score_labels.append(score)
        rl.addStretch()
        note = label('Skorlar modelin değerlendirmesidir; tıbbi doğruluk olasılığı değildir.', 'footnote')
        note.setWordWrap(True)
        rl.addWidget(note)
        self.export_button = QPushButton('Sonucu kaydet…')
        self.export_button.clicked.connect(self.export_result)
        rl.addWidget(self.export_button)
        body.addWidget(result_card, 4)
        page.addLayout(body, 1)
        footer = label(f'Bu sürüm {AUTHOR} tarafından geliştirilmiştir.\nAraştırma amaçlı prototip • Klinik tanı için doğrulanmamıştır. Sonuçlar uzman değerlendirmesinin yerine geçmez.', 'footnote')
        footer.setWordWrap(True)
        page.addWidget(footer)
        self.refresh_models()
        self.update_controls()
        if model is not None:
            self.model_ready(model)
        elif auto_load and self.model_combo.count() and self.model_combo.currentData():
            self.load_model(self.model_combo.currentData())

    def refresh_models(self, selected=None):
        paths = sorted((ROOT/'models').glob('*.pt'))
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for path in paths:
            name = 'EfficientNet-B0 • Yerel model' if path.name == 'aptos_efficientnet_b0.pt' else path.stem.replace('_',' ')
            self.model_combo.addItem(name, str(path))
        if not paths: self.model_combo.addItem('Henüz model eklenmedi', None)
        if selected:
            index = self.model_combo.findData(str(Path(selected)))
            if index >= 0: self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)

    def update_controls(self):
        idle = not self.busy
        self.button.setEnabled(idle)
        self.import_button.setEnabled(idle)
        self.model_combo.setEnabled(idle)
        self.clear_button.setEnabled(idle and self.image_path is not None)
        self.analyze_button.setEnabled(idle and self.model is not None and self.image_path is not None)
        self.export_button.setEnabled(idle and self.last_result is not None)
        self.progress.setVisible(self.busy)

    def reset_result(self):
        self.last_result = None
        self.result_tag.setText('ANALİZ BEKLENİYOR')
        self.result_tag.setStyleSheet('')
        self.result_title.setText('Henüz sonuç yok')
        self.status.setText('Analizi başlattığınızda model değerlendirmesi burada görünecek.')
        for bar, score in zip(self.bars, self.score_labels):
            bar.setValue(0)
            score.setText('—')
        self.update_controls()

    def select_model(self, index):
        path = self.model_combo.itemData(index)
        if path: self.load_model(path)

    def load_model(self, path, do_import=False):
        if self.busy: return
        self.reset_result()
        self.busy = True
        self.badge.setText('Model yükleniyor')
        self.model_info.setText('Model kontrol ediliyor, lütfen bekleyin…')
        self.update_controls()
        self.model_worker = ModelThread(path, do_import, self)
        self.model_worker.ready.connect(self.model_ready)
        self.model_worker.failed.connect(self.model_failed)
        self.model_worker.finished.connect(self.work_finished)
        self.model_worker.finished.connect(self.model_worker.deleteLater)
        self.model_worker.start()

    def model_ready(self, model):
        self.model = model
        path = getattr(model, 'checkpoint_path', None)
        self.refresh_models(path)
        name = getattr(model, 'display_name', 'Eğitilmiş model')
        self.model_info.setText(f'{name} • Belirsizlik eşiği {percentage(model.threshold)}')
        self.badge.setText('Model hazır')
        self.update_controls()

    def model_failed(self, message):
        if self.model is not None:
            self.model_ready(self.model)
        else:
            self.badge.setText('Model yüklenemedi')
            self.model_info.setText('Uyumlu bir model ekleyerek devam edin.')
        self.failed('Model açılamadı. ' + message)

    def work_finished(self):
        worker = self.sender()
        if worker is self.worker: self.worker = None
        if worker is self.model_worker: self.model_worker = None
        self.busy = False
        self.update_controls()

    def import_model(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Eğitilmiş model ekle', str(ROOT/'models'), 'Uyumlu PyTorch modeli (*.pt)')
        if path: self.load_model(path, do_import=True)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Retina görüntüsü seç', '', 'Görüntüler (*.png *.jpg *.jpeg *.bmp)')
        if path: self.set_image(path)

    def set_image(self, path):
        if self.busy: return False
        path = Path(path)
        pix = QPixmap(str(path))
        if path.suffix.lower() not in FORMATS or pix.isNull():
            self.failed('Görüntü okunamadı. PNG, JPG veya BMP biçiminde bir dosya seçin.')
            return False
        self.image_path = path
        self.original_pixmap = pix
        self.file_info.setText(f'{path.name}\n{pix.width()} × {pix.height()} piksel')
        self.reset_result()
        self.render_preview()
        self.update_controls()
        return True

    def render_preview(self):
        if self.original_pixmap:
            self.preview.setPixmap(self.original_pixmap.scaled(self.preview.size()-QSize(18,18), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'preview'): self.render_preview()

    def clear_image(self):
        if self.busy: return
        self.image_path, self.original_pixmap = None, None
        self.preview.clear()
        self.preview.setText('Retina görüntüsünü buraya sürükleyin\nveya aşağıdaki düğmeyle seçin.')
        self.file_info.setText('PNG, JPG veya BMP • Görüntüler bu bilgisayarda işlenir.')
        self.reset_result()

    def analyze(self):
        if self.busy or self.model is None or self.image_path is None: return
        self.reset_result()
        self.busy = True
        self.result_tag.setText('ANALİZ SÜRÜYOR')
        self.result_title.setText('Görüntü inceleniyor')
        self.status.setText('İşlem bu bilgisayarda yürütülüyor. Lütfen bekleyin.')
        self.update_controls()
        self.worker = PredictThread(self.model, str(self.image_path), self)
        self.worker.result.connect(self.show_result)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.work_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def show_result(self, result):
        self.last_result = result
        top = result['top_class']
        score = result.get('score', max(result['scores']))
        threshold = getattr(self.model, 'threshold', .8)
        if result['class_id'] < 0:
            self.result_tag.setText('BELİRSİZ DEĞERLENDİRME')
            self.result_tag.setStyleSheet('color: #996214;')
            self.result_title.setText('Sonuç belirsiz')
            self.status.setText(f'Model en yüksek skoru “{LABELS[top]}” sınıfına verdi ({percentage(score)}). '
                                f'Skor, {percentage(threshold)} eşiğinin altında; bir sınıf sonucu sunulmuyor.')
        else:
            self.result_tag.setText('MODELİN ÖNGÖRDÜĞÜ SINIF')
            self.result_tag.setStyleSheet('')
            self.result_title.setText(LABELS[result['class_id']])
            self.status.setText(f'En yüksek model skoru: {percentage(score)}. Bu değerlendirme tek başına tanı koymaz.')
        for bar, widget, value in zip(self.bars, self.score_labels, result['scores']):
            bar.setValue(round(value*100))
            widget.setText(percentage(value))
        self.update_controls()

    def failed(self, message):
        self.last_result = None
        self.result_tag.setText('İŞLEM TAMAMLANAMADI')
        self.result_title.setText('Lütfen kontrol edin')
        self.status.setText(message)
        for bar, score in zip(self.bars, self.score_labels):
            bar.setValue(0); score.setText('—')
        self.update_controls()

    def export_result(self):
        if self.last_result is None or self.image_path is None: return
        path, _ = QFileDialog.getSaveFileName(self, 'Analiz sonucunu kaydet', f'{self.image_path.stem}_analiz.json', 'JSON raporu (*.json)')
        if not path: return
        payload = {'application': 'Blindless Detection', 'application_developer': AUTHOR,
                   'created_at': datetime.now().astimezone().isoformat(), 'image': self.image_path.name,
                   'model': getattr(self.model,'display_name',''), 'checkpoint': getattr(self.model,'checkpoint_path',Path('')).name,
                   'threshold': getattr(self.model,'threshold',.8), 'classes': LABELS,
                   'result': self.last_result, 'purpose':'Araştırma amaçlıdır; klinik tanı için doğrulanmamıştır.'}
        try:
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            self.status.setText(self.status.text() + '\nRapor kaydedildi.')
        except OSError as exc:
            QMessageBox.warning(self, 'Rapor kaydedilemedi', str(exc))

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if not self.busy and len(urls)==1 and urls[0].isLocalFile() and Path(urls[0].toLocalFile()).suffix.lower() in FORMATS:
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if len(urls)==1 and self.set_image(urls[0].toLocalFile()): event.acceptProposedAction()

    def closeEvent(self, event):
        if self.busy:
            event.ignore()
            self.status.setText('Devam eden işlem tamamlandıktan sonra pencereyi kapatabilirsiniz.')
        else: event.accept()

    def reject(self):
        # Escape uses QDialog.reject directly, bypassing closeEvent.
        if self.busy:
            self.status.setText('Devam eden işlem tamamlandıktan sonra pencereyi kapatabilirsiniz.')
            return
        super().reject()
