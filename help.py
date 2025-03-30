import os

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, 
                            QWidget, QFrame, QLabel)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl


class Help(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.help_window = None
        self.html_paths = {
            1: 'html/en/generate_pure_tone/Generatepuretone.html',
            2: 'html/en/harmonic_synthesis/Harmonicsynthesis.html',
            3: 'html/en/generate_square_wave/Generatesquarewave.html',
            4: 'html/en/generate_sawtooth_signal/Generatesawtoothsignal.html',
            5: 'html/en/generate_noise/Generatenoise.html',
            6: 'html/en/load_audio_file/Loadaudiofile.html',
            7: 'html/en/record_audio/Recordaudio.html',
            8: 'html/en/visualization_window/Visualizationwindow.html'
        }



    def createHelpMenu(self, value):
        if self.help_window is None:
            self.help_window = QDialog(self)
            self.help_window.setWindowTitle('Help Menu')
            self.help_window.setWindowFlags(self.help_window.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            self.help_window.resize(975, 600)
            
            # Main layout
            main_layout = QHBoxLayout()
            main_layout.setContentsMargins(5, 5, 5, 5)
            
            # Left panel for radio buttons
            left_panel = QFrame()
            left_layout = QVBoxLayout()
            left_layout.setSpacing(5)
            
            # Create radio buttons
            self.radio_group = []
            for i in range(1, 9):
                radio = QRadioButton(self.get_button_text(i))
                radio.setChecked(i == value)
                radio.clicked.connect(self.create_radio_handler(i))
                left_layout.addWidget(radio)
                self.radio_group.append(radio)
            
            left_panel.setLayout(left_layout)
            main_layout.addWidget(left_panel)
            
            # Right panel for HTML content
            self.web_view = QWebEngineView()
            main_layout.addWidget(self.web_view, 1)
            
            self.help_window.setLayout(main_layout)
            self.help_window.finished.connect(self.on_help_close)
        
        # Show initial content
        self.show_help(value)
        self.help_window.show()
        self.help_window.raise_()
        self.help_window.activateWindow()

    def on_help_close(self):    
        """Clean up when help window is closed"""
        self.help_window = None
        print("Help window closed")  # Optional debug output

    def create_radio_handler(self, value):
        def handler():
            print(f"Loading help page {value}")
            self.show_help(value)
        return handler

    def show_help(self, value):
        if not self.help_window:  # Safety check
            return
        
        print(f"Loading help page {value}")  # Debug output
        
        html_file = self.html_paths.get(value)
        if html_file:
            try:
                # Convert to absolute path
                abs_path = os.path.abspath(html_file)
                base_dir = os.path.dirname(abs_path)
                
                # Read HTML content
                with open(html_file, 'r', encoding='utf-8') as file:
                    html_content = file.read()
                
                # Set base URL for relative paths
                base_url = QUrl.fromLocalFile(base_dir + '/')
                self.web_view.setHtml(html_content, base_url)
                
                # Update radio button selection
                for i, radio in enumerate(self.radio_group, start=1):
                    radio.setChecked(i == value)
                    
            except Exception as e:
                print(f"Error loading help content: {str(e)}")
                self.web_view.setHtml(f"<h1>Error</h1><p>Could not load help page: {str(e)}</p>")

    def get_button_text(self, value):
        """Returns the display text for each help section"""
        help_topics = {
            1: "Pure Tone",
            2: "Harmonic Synthesis",
            3: "Square Wave",
            4: "Sawtooth Wave", 
            5: "Noise Generation",
            6: "Load Audio File",
            7: "Record Audio",
            8: "Visualization"
        }
        return help_topics.get(value, f"Help {value}")