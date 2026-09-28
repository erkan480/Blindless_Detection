import sqlite3
from pathlib import Path

from PyQt5.QtWidgets import QPushButton

# gerekli modüller import ediliyor.

icon_folder = (Path(__file__).resolve().parents[2] / "icons").as_posix() + "/" # iconların bulunduğu klasör

resources_folder = "src/resources/" # kaynak dosyalarının bulunduğu klasör

# style sheet fonksiyonu
get_features = lambda font="Kantumruy", size=17, color="black",background_color="transparent",\
                      border=0,border_color="black":("font-family: {}; font-size: {}px; color: {};background-color: {};border: {}px solid {};"
     .format(font, size, color, background_color,border,border_color))


def set_checkbox_icon(checkbox,path,x=100,y=100): # check_box icon için fonksiyon
    checkbox.setStyleSheet(f'''
        QCheckBox::indicator {{
            width: {x}px;
            height: {y}px;
            border-image: url({path});
        }}
    ''')

class RoundButton(QPushButton): # özelleştirilmiş buton
    def __init__(self,x=250,y=60,text="Go"):
        super().__init__()
        #border-radius: {y//2}px;
        self.setFixedSize(x,y)
        self.setText(text)
        self.setStyleSheet(f"""
    QPushButton {{
        background-color: grey;
        color: black;
        font-family: {"Kantumruy"};
        font-size: {30}px;
        
    }}
    QPushButton:hover {{
        background-color: #45a049;
        color : white;
    }}
""")


# widget özelleştirme
def customize_widget(widget, text_size = 40, background_color = "transparent",
                     color = "black",border_color = "black", border = 0, text="", font = "Kantumruy"):
    widget.adjustSize()
    widget.setStyleSheet(
            get_features(size=text_size, background_color=background_color, color=color,
                         border=border, border_color=border_color,font=font))
    widget.setText(text)

# pencerelerde göstermek için dosya ismini sadeleştiriyor.
def get_file_name(path):
    last_index = path.rfind("/") # son '/' karakteri
    filename = path[last_index+1:-4] # dosya uzantısı çıkarılıyor.
    return filename
    # örneğin : C:Users/Desktop/....resources/comments.txt --> ismi : comments olarak döndürür.