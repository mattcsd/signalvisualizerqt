from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QFrame, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class Info(QWidget):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.master = master
        self.initUI()
        
    def initUI(self):
        # Create a scroll area to handle long content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        # Main container widget
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Signal Visualizer")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Introduction
        intro = QLabel("A collaboration between University of the Basque Country (UPV/EHU)\n"
                      "and Musikene, Higher School of Music of the Basque Country.")
        layout.addWidget(intro)
        layout.addSpacing(10)
        
        # Team information
        self.add_label_pair(layout, "Project leader:", "Inma Hernáez Rioja")
        self.add_label_pair(layout, "Project leader from Musikene:", "José María Bretos")
        self.add_label_pair(layout, "Developers:", "Leire Varela Aranguren, Valentin Lurier, Eder del Blanco Sierra, Mikel Díez García")
        self.add_label_pair(layout, "Program icon and logo made by:", "Sergio Santamaría Martínez")
        
        # Add spacing
        layout.addSpacing(20)
        
        # Aholab information
        aholab = QLabel("HiTZ Basque Center for Language Technologies - Aholab Signal Processing Laboratory (UPV/EHU).")
        layout.addWidget(aholab)
        
        # References title
        ref_title = QLabel("References:")
        ref_title.setFont(title_font)
        layout.addWidget(ref_title)
        
        # References content
        ref1 = QLabel(
            "Function for creating brown (or red) noise made by Hristo Zhivomirov:\n"
            "Hristo Zhivomirov (2020). Pink, Red, Blue and Violet Noise Generation with Matlab.\n"
            "https://www.mathworks.com/matlabcentral/fileexchange/42919-pink-red-blue-and-violet-noise-generation-with-matlab\n"
            "MATLAB Central File Exchange. Retrieved August 4, 2020."
        )
        ref1.setWordWrap(True)
        layout.addWidget(ref1)
        
        ref2 = QLabel(
            "Master thesis describing the version of the software Signal Visualizer in Matlab made by Eder del Blanco Sierra:\n"
            "Eder del Blanco Sierra (2020). Programa de apoyo a la enseñanza musical.\n"
            "University of the Basque Country (UPV/EHU). Department of Communications Engineering. Retrieved August 8, 2020.\n"
            "The function has been modified by Valentin Lurier and Mikel Díez García."
        )
        ref2.setWordWrap(True)
        layout.addWidget(ref2)
        
        # Add stretch to push content up
        layout.addStretch()
        
        # Set the scroll area's widget
        scroll.setWidget(container)
        
        # Main layout for this widget
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
    
    def add_label_pair(self, layout, label_text, value_text):
        """Helper method to add label-value pairs with consistent styling"""
        container = QWidget()
        h_layout = QVBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(label_text)
        label_font = QFont()
        label_font.setItalic(True)
        label.setFont(label_font)
        
        value = QLabel(value_text)
        
        h_layout.addWidget(label)
        h_layout.addWidget(value)
        layout.addWidget(container)