from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QPushButton

class Settings(QWidget):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        
        # Setup UI
        layout = QVBoxLayout()
        
        # Font size control
        self.font_spin = QSpinBox()
        self.font_spin.setRange(8, 24)
        self.font_spin.setValue(self.main_window.app_settings['font_size'])
        
        # Apply button
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self.apply_settings)
        
        layout.addWidget(QLabel("Font Size:"))
        layout.addWidget(self.font_spin)
        layout.addWidget(btn_apply)
        self.setLayout(layout)
    
    def apply_settings(self):
        """Update settings and notify main window"""
        new_size = self.font_spin.value()
        self.main_window.app_settings['font_size'] = new_size
        self.main_window.settings_changed.emit(self.main_window.app_settings)