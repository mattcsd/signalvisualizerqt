from PyQt5.QtWidgets import (QSizePolicy, QWidget, QDialog, QLabel, QVBoxLayout, QPushButton, 
                             QScrollArea, QGroupBox, QCheckBox)
from PyQt5.QtCore import QSettings, Qt

class FirstRunDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to Signal Visualizer")
        self.resize(600, 500)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Welcome message - fixed at top (non-scrolling)
        welcome_label = QLabel("""
            <div style="
                background-color: #f0f8ff;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #d3d3d3;
                margin-bottom: 10px;
            ">
                <h1 style="color: #2c3e50; margin-top: 0; text-align: center;">
                    Welcome to Signal Visualizer <span style="font-size: 0.7em;">(Beta Version)</span>
                </h1>
                
                <div style="background-color: white; padding: 12px; border-radius: 6px; margin: 10px 0;">
                    <h3 style="color: #3498db; margin-top: 0;">Some notes:</h3>
                    <ul style="margin: 5px 0; padding-left: 25px;">
                        <li style="margin-bottom: 8px;"><b>Waveform interaction:</b> Click-hold and select a span to play any waveform</li>
                        <li style="margin-bottom: 8px;"><b>Save function:</b> Currently disabled (coming in final release)</li>
                        <li style="margin-bottom: 8px;"><b>Help pages/examples explanation:</b> Will be added when we reach final version</li>
                        <li style="margin-bottom: 8px;"><b>Examples page:</b> If you go full screen, click "Replot" to fix the GUI</li>
                        <li style="margin-bottom: 8px;"><b>If in generator</b> the signal seem short so you can visualize them, set duration from below</li>
                        <li><b>Live analysis:</b> STFT+Spectrogram live analysis is still under development</li>
                    </ul>
                </div>
                <p style="font-style: italic; color: #7f8c8d; text-align: center; margin-bottom: 0;">
                    Thank you for testing our beta version! Your feedback is valuable.
                </p>
            </div>
        """)
        welcome_label.setWordWrap(True)
        main_layout.addWidget(welcome_label)

        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(15)

        # Menu explanations
        self.add_menu_section(content_layout, "Signal Visualizer Menu", [
            "Info: View information about the application",
            "Exit: Close the program"
        ])
        

        self.add_menu_section(content_layout, "Generate Menu", [
            "Pure tone: Generate a single frequency tone",
            "Free addition of pure tones: Combine multiple tones",
            "Noise: Generate different types of noise signals",
            "Known periodic signals → Square wave: Generate square wave signals",
            "Known periodic signals → Sawtooth wave: Generate sawtooth wave signals"
        ], 
        intro_text="The Generate menu contains tools for creating various types of audio signals. ")

        self.add_menu_section(content_layout, "Input Menu", [
            "Load: Load audio files from your computer",
            "Record: Record audio from your microphone"
        ])
        
        self.add_menu_section(content_layout, "Tuner Menu", [
            "Live STFT: Real-time audio frequency analysis"
        ])
        
        self.add_menu_section(content_layout, "Examples Menu", [
            "Cretan Lute: Example of some recordings i made to analyse what i found interesting concepts"
        ])
        
        self.add_menu_section(content_layout, "Options Menu", [
            "Spectrogram: Configure spectrogram settings"
        ])

        # Important: Set size policy and minimum size
        content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_widget.setMinimumSize(500, 400)  # Set minimum size to ensure scroll appears

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        # Bottom controls
        self.dont_show_checkbox = QCheckBox("Don't show this again")
        main_layout.addWidget(self.dont_show_checkbox)

        close_button = QPushButton("Get Started")
        close_button.clicked.connect(self.accept)
        main_layout.addWidget(close_button)
    
    def add_menu_section(self, layout, title, items, intro_text=None):
        """Helper method to add a menu section with optional intro text"""
        group = QGroupBox(title)
        group_layout = QVBoxLayout()
        
        if intro_text:
            intro_label = QLabel(intro_text)
            intro_label.setWordWrap(True)
            intro_label.setStyleSheet("font-style: italic; margin-bottom: 10px;")
            group_layout.addWidget(intro_label)
        
        for item in items:
            label = QLabel(f"• {item}")
            label.setWordWrap(True)
            group_layout.addWidget(label)
        
        group.setLayout(group_layout)
        layout.addWidget(group)