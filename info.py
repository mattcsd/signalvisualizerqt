from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QFrame)
from PyQt5.QtCore import Qt

class Info(QWidget):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.master = master
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        
        # Create labels with the information
        title = QLabel("Signal Visualizer")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        
        intro = QLabel("A collaboration between University of the Basque Country (UPV/EHU)\nand Musikene, Higher School of Music of the Basque Country.")
        
        # Create a frame for the people information
        people_frame = QFrame()
        people_layout = QVBoxLayout()
        
        def create_label_pair(role, name):
            role_label = QLabel(role)
            role_label.setStyleSheet("font-style: italic;")
            name_label = QLabel(name)
            return role_label, name_label
            
        leader_role, leader_name = create_label_pair("Project leader:", "Inma Hernáez Rioja")
        contact_role, contact_name = create_label_pair("Project leader from Musikene:", "José María Bretos")
        dev_role, dev_name = create_label_pair("Developers:", "Matteo Tsikalakis Reeder, Leire Varela Aranguren, Valentin Lurier, Eder del Blanco Sierra, Mikel Díez García")
        icon_role, icon_name = create_label_pair("Program icon and logo made by:", "Sergio Santamaría Martínez")
        
        for widget in [leader_role, leader_name, contact_role, contact_name, 
                      dev_role, dev_name, icon_role, icon_name]:
            people_layout.addWidget(widget)
            
        people_frame.setLayout(people_layout)
        
        # References section
        aholab = QLabel("\nHiTZ Basque Center for Language Technologies - Aholab Signal Processing Laboratory (UPV/EHU).\n")
        references = QLabel("References:")
        ref1 = QLabel("Function for creating brown (or red) noise made by Hristo Zhivomirov:\nHristo Zhivomirov (2020). Pink, Red, Blue and Violet Noise Generation with Matlab.\nhttps://www.mathworks.com/matlabcentral/fileexchange/42919-pink-red-blue-and-violet-noise-generation-with-matlab\nMATLAB Central File Exchange. Retrieved August 4, 2020.")
        ref2 = QLabel("Master thesis describing the version of the software Signal Visualizer in Matlab made by Eder del Blanco Sierra:\nEder del Blanco Sierra (2020). Programa de apoyo a la enseñanza musical.\nUniversity of the Basque Country (UPV/EHU). Department of Communications Engineering. Retrieved August 8, 2020.\nThe function has been modified by Valentin Lurier and Mikel Díez García.")
        
        # Add all widgets to the main layout
        for widget in [title, intro, people_frame, aholab, references, ref1, ref2]:
            layout.addWidget(widget)
            
        self.setLayout(layout)