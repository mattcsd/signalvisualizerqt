import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QMenuBar, QMenu, QAction, QMessageBox, QDesktopWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from info import Info
from help import Help
from inputLoad import Load
from inputRecord import Record
from generateNoise import Noise
from generatePureTone import PureTone
from generateFreeAdd import FreeAdditionPureTones
from generateSquareWave import SquareWave
from generateSawtoothWave import SawtoothWave
from optionsSpectrogram import Spectrogram
from pitchAdvancedSettings import AdvancedSettings
from auxiliar import Auxiliar

import matplotlib.pyplot as plt
from matplotlib import backend_bases
from pitchAdvancedSettings import PitchAdvancedSettingsHandler

# To avoid blurry fonts on Windows
if sys.platform == "win32":
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)

# If in mac the menu bar is not at the top as by default but within the window.
if sys.platform == "darwin":  # macOS
    QApplication.setAttribute(Qt.AA_DontUseNativeMenuBar)


class Start(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Visualizer")
        self.setWindowIcon(QIcon('icons/icon.ico'))

        

        self.icons = {
            'icon': QIcon('icons/icon.ico')  # Make sure this path is correct
        }

        # Set window size and center it
        self.resize(900, 650)  # Increased size to accommodate Info content
        self.center_window()    # Center the window

        # Central widget and layout
        self.container = QWidget()
        self.setCentralWidget(self.container)
        self.layout = QVBoxLayout(self.container)
        self.container.setLayout(self.layout)

        # Dictionary to hold frames
        self.frames = {}
        
        # Initialize and show Info frame by default
        self.initialize_frame('Info')

        # Initialize Help and AdvancedSettings
        self.help = Help(self.container, self)
        self.adse = PitchAdvancedSettingsHandler(parent=self)
        # Set up the menu bar
        self.create_menu_bar()

    def center_window(self):
        """Center the window on the screen."""
        screen = QDesktopWidget().screenGeometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def initialize_frame(self, page_name):
        """Initialize and display a frame based on the page name."""
        if page_name == 'SignalVisualizer':
            self.frames['SignalVisualizer'] = SignalVisualizer(self.container, self)
        elif page_name == 'Info':
            self.frames['Info'] = Info(self.container, self)
        elif page_name == 'Load':
            self.frames['Load'] = Load(self.container, self)
        elif page_name == 'Record':
            self.frames['Record'] = Record(self.container, self)
        elif page_name == 'Noise':
            self.frames['Noise'] = Noise(self.container, self)
        elif page_name == 'PureTone':
            self.frames['PureTone'] = PureTone(self.container, self)
        elif page_name == 'FreeAdditionPureTones':
            self.frames['FreeAdditionPureTones'] = FreeAdditionPureTones(self.container, self)
        elif page_name == 'SquareWave':
            self.frames['SquareWave'] = SquareWave(self.container, self)
        elif page_name == 'SawtoothWave':
            self.frames['SawtoothWave'] = SawtoothWave(self.container, self)
        elif page_name == 'Spectrogram':
            self.frames['Spectrogram'] = Spectrogram(self.container, self)

        # Show the frame
        self.show_frame(page_name)

    def show_frame(self, page_name):
        """Show the frame corresponding to the given page name."""
        if page_name in self.frames:
            # Remove the current widget from the layout
            for i in reversed(range(self.layout.count())):
                self.layout.itemAt(i).widget().setParent(None)

            # Add the new frame to the layout
            self.layout.addWidget(self.frames[page_name])
            self.frames[page_name].setVisible(True)

    def create_menu_bar(self):
        """Create the menu bar for the application."""
        menubar = self.menuBar()

        # Signal Visualizer menu
        signal_menu = menubar.addMenu("Signal Visualizer")
        signal_menu.addAction("Info", lambda: self.initialize_frame('Info'))
        signal_menu.addAction("Exit", self.close)

        # Generate menu
        generate_menu = menubar.addMenu("Generate")
        generate_menu.addAction("Pure tone", lambda: self.initialize_frame('PureTone'))
        generate_menu.addAction("Free addition of pure tones", lambda: self.initialize_frame('FreeAdditionPureTones'))
        generate_menu.addAction("Noise", lambda: self.initialize_frame('Noise'))

        # Known periodic signals submenu
        known_menu = generate_menu.addMenu("Known periodic signals")
        known_menu.addAction("Square wave", lambda: self.initialize_frame('SquareWave'))
        known_menu.addAction("Sawtooth wave", lambda: self.initialize_frame('SawtoothWave'))

        # Input menu
        input_menu = menubar.addMenu("Input")
        input_menu.addAction("Load", lambda: self.initialize_frame('Load'))
        input_menu.addAction("Record", lambda: self.initialize_frame('Record'))

        # Options menu
        options_menu = menubar.addMenu("Options")
        options_menu.addAction("Spectrogram", lambda: self.initialize_frame('Spectrogram'))

    def closeEvent(self, event):
        """Handle the window close event."""
        reply = QMessageBox.question(
            self,
            "Quit",
            "Do you want to quit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            plt.close('all')  # Close all matplotlib figures
            event.accept()
        else:
            event.ignore()


class SignalVisualizer(QWidget):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.master = master


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Start()
    window.show()
    sys.exit(app.exec_())