from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QRadioButton, QButtonGroup
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl

import sys
import os   

class Help(QMainWindow):
    def __init__(self, master, value=1):
        super().__init__()
        self.setWindowTitle("Help Menu")
        self.setGeometry(100, 100, 975, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.web_view = QWebEngineView()
        
        self.button_group = QButtonGroup(self)
        self.buttons = {
            1: ("Pure Tone", "html/en/Generate noise/Generatenoise.html"),
            2: ("Addition of tones", "html/en/Harmonic synthesis/Harmonicsynthesis.html"),
            3: ("Square wave", "html/en/Generate square wave/Generatesquarewave.html"),
            4: ("Sawtooth wave", "html/en/Generate sawtooth signal/Generatesawtoothsignal.html"),
            5: ("Noise", "html/en/Generate noise/Generatenoise.html"),
            6: ("Load file", "html/en/Load audio file/Loadaudiofile.html"),
            7: ("Record sound", "html/en/Record audio/Recordaudio.html"),
            8: ("Control menu", "html/en/Visualization window/Visualizationwindow.html"),
        }
        
        for value, (label, html_path) in self.buttons.items():
            btn = QRadioButton(label)
            btn.clicked.connect(lambda checked, p=html_path: self.load_html(p))
            self.layout.addWidget(btn)
            self.button_group.addButton(btn, value)
            
        self.layout.addWidget(self.web_view)
        self.load_html(self.buttons[value][1])  # Load initial page
        
    def load_html(self, html_path):
        abs_path = os.path.abspath(html_path)  # Convert to absolute path
        self.web_view.setUrl(QUrl.fromLocalFile(abs_path))  # Use QUrl properly

if __name__ == "__main__":
    app = QApplication(sys.argv)
    help_window = Help(1)  # Default to first option
    help_window.show()
    sys.exit(app.exec_())
