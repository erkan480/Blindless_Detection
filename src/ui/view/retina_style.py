"""Desktop presentation tokens; no model behavior lives here."""
STYLE = '''
QWidget { font-family: "Segoe UI"; font-size: 13px; color: #23354b; }
QDialog#root { background: #eef3f7; }
QScrollArea, QScrollArea > QWidget > QWidget { background: #eef3f7; border: none; }
QFrame#card { background: white; border: 1px solid #dce5ec; border-radius: 14px; }
QLabel { background: transparent; border: none; }
QLabel#brand { background: #167c80; color: white; border-radius: 12px; font-size: 20px; font-weight: 700; }
QLabel#title { font-size: 25px; font-weight: 700; color: #142b43; }
QLabel#section { font-size: 16px; font-weight: 600; color: #142b43; }
QLabel#muted { color: #66778b; }
QLabel#resultTitle { font-size: 23px; font-weight: 600; color: #142b43; }
QLabel#eyebrow { color: #167c80; font-size: 11px; font-weight: 700; }
QLabel#preview { background: #f5f8fa; border: 1px dashed #b8cbd5; border-radius: 10px; color: #66778b; }
QLabel#badge { background: #e5f3f0; color: #11685f; padding: 5px 10px; border-radius: 10px; font-size: 12px; }
QLabel#footnote { color: #6b7b8b; font-size: 11px; }
QPushButton { background: white; border: 1px solid #cbd8e0; border-radius: 7px; padding: 9px 14px; font-weight: 600; }
QPushButton:hover { background: #edf5f6; border-color: #167c80; }
QPushButton:disabled { background: #eef2f5; border-color: #e0e7ec; color: #97a3af; }
QPushButton#primary { background: #167c80; color: white; border: 1px solid #167c80; }
QPushButton#primary:hover { background: #11686c; }
QPushButton#primary:disabled { background: #c5d8da; border-color: #c5d8da; color: #f5f8fa; }
QComboBox { background: #f8fafc; border: 1px solid #cbd8e0; border-radius: 7px; padding: 8px 12px; min-height: 18px; }
QComboBox QAbstractItemView { background: white; selection-background-color: #d9eeee; selection-color: #142b43; }
QProgressBar { background: #eaf0f4; border: none; border-radius: 4px; min-height: 8px; max-height: 8px; }
QProgressBar::chunk { background: #26969a; border-radius: 4px; }
QToolTip { background: #142b43; color: white; border: none; padding: 6px; }
'''
