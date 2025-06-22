import sys
from PyQt5.QtWidgets import QToolButton, QWidgetAction, QHBoxLayout, QAction, QApplication, QMenuBar, QMenu, QMainWindow, QWidget, QVBoxLayout, QMenuBar, QMenu, QAction, QMessageBox, QDesktopWidget
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
from simpletuner import AudioFFTVisualizer
from examples import BeatFrequencyVisualizer

import matplotlib.pyplot as plt
from matplotlib import backend_bases
from pitchAdvancedSettings import PitchAdvancedSettingsHandler
from popupinfo import FirstRunDialog

from fundamentalSeparator import FundamentalHarmonicsSeparator


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
        self.show_welcome_dialog()

        # Initialize Help and AdvancedSettings
        self.help = Help(self.container, self)
        self.adse = PitchAdvancedSettingsHandler(parent=self)
        # Set up the menu bar
        self.create_menu_bar()


    def show_welcome_dialog(self):
        dialog = FirstRunDialog(self)
        dialog.exec_()

    def center_window(self):
        """Center the window on the screen."""
        screen = QDesktopWidget().screenGeometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def initialize_frame(self, page_name):
        """Initialize and display a frame based on the page name."""
        # Clean up existing frame if it exists
        if page_name in self.frames and hasattr(self.frames[page_name], 'cleanup'):
            self.frames[page_name].cleanup()
        
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
        elif page_name == 'Tuner':
            self.frames['Tuner'] = AudioFFTVisualizer(self.container, self)
        elif page_name == 'Cretan Lute':
            self.frames['Cretan Lute'] = BeatFrequencyVisualizer(self.container, self)
                # Show the frame
        self.show_frame(page_name)

    def show_frame(self, page_name):
        """Show the frame corresponding to the given page name."""
        if page_name in self.frames:
            # Clean up current frame if it exists and has cleanup method
            current_widget = self.layout.itemAt(0).widget() if self.layout.count() > 0 else None
            if current_widget and hasattr(current_widget, 'cleanup'):
                current_widget.cleanup()
            
            # Remove the current widget from the layout
            for i in reversed(range(self.layout.count())):
                widget = self.layout.itemAt(i).widget()
                widget.setParent(None)
                if hasattr(widget, 'cleanup'):  # Additional safety check
                    widget.cleanup()
            
            # Add the new frame to the layout
            self.layout.addWidget(self.frames[page_name])
            self.frames[page_name].setVisible(True)


    def create_menu_bar(self):
        """Create the menu bar with button-style dropdown items that support tooltips"""
        menubar = self.menuBar()
        
        # Keep your existing stylesheet for the menu bar itself
        menubar.setStyleSheet("""
            /* Main menu bar */
            QMenuBar {
                background-color: #2c3e50;
                color: white;
                font-size: 1em;
                font-weight: bold;
                padding: 0.5em;
                spacing: 0.5em;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 0.5em 1em;
                border-radius: 0.25em;
            }
            QMenuBar::item:selected {
                background-color: #3498db;
            }
            QMenuBar::item:pressed {
                background-color: #2980b9;
            }
            
            /* Dropdown menus */
            QMenu {
                background-color: #34495e;
                color: white;
                border: 1px solid #555;
                padding: 0.25em;
                font-size: 1em;
                min-width: 12em;  /* Based on typical character width */
            }
            
            /* Menu buttons */
            QToolButton {
                background-color: transparent;
                color: white;
                border: none;
                padding: 0.75em 1.5em;
                text-align: left;
                min-width: 12em;
                min-height: 2.25em;
                font-size: 1em;
            }
            QToolButton:hover {
                background-color: #3498db;
                border-radius: 0.2em;
            }
            
            /* Tooltips */
            QToolTip {
                background-color: #34495e;
                color: white;
                border: 1px solid #3498db;
                padding: 0.5em;
                border-radius: 0.25em;
                font-size: 18pt;
                opacity: 230;
            }
        """)
        # Signal Visualizer menu
        signal_menu = menubar.addMenu("Signal Visualizer")
        self._add_menu_button(signal_menu, "Info", "Show application information and instructions", 
                            lambda: self.initialize_frame('Info'))
        self._add_menu_button(signal_menu, "Exit", "Exit the application", self.close)

        # Generate menu
        generate_menu = menubar.addMenu("Generate")
        self._add_menu_button(generate_menu, "Pure tone", "Generate a single frequency sine wave",
                             lambda: self.initialize_frame('PureTone'))
        self._add_menu_button(generate_menu, "Free addition of pure tones", 
                            "Combine multiple sine waves with custom frequencies",
                            lambda: self.initialize_frame('FreeAdditionPureTones'))
        self._add_menu_button(generate_menu, "Noise", "Generate different types of noise signals",
                             lambda: self.initialize_frame('Noise'))

        # Known periodic signals submenu
        known_menu = generate_menu.addMenu("Known periodic signals")
        self._add_menu_button(known_menu, "Square wave", "Generate a square wave signal",
                             lambda: self.initialize_frame('SquareWave'))
        self._add_menu_button(known_menu, "Sawtooth wave", "Generate a sawtooth wave signal",
                             lambda: self.initialize_frame('SawtoothWave'))

        # Input menu
        input_menu = menubar.addMenu("Input")
        self._add_menu_button(input_menu, "Load", "Load an audio file from disk",
                             lambda: self.initialize_frame('Load'))
        self._add_menu_button(input_menu, "Record", "Record audio from your microphone",
                             lambda: self.initialize_frame('Record'))

        # Tuner menu
        tuner_menu = menubar.addMenu("Tuner")
        self._add_menu_button(tuner_menu, "Live STFT", "Real-time frequency analysis for tuning instruments",
                             lambda: self.initialize_frame('Tuner'))

        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        self._add_menu_button(tools_menu, "Fundamental/Harmonics Separator",
                            "Separate fundamental frequency from harmonics",
                            lambda: self.show_separator_tool())

        # Examples menu
        examples_menu = menubar.addMenu("Examples")
        self._add_menu_button(examples_menu, "Cretan Lute", "Example analysis of Cretan Lute audio",
                             lambda: self.initialize_frame('Cretan Lute'))

        # Options menu
        options_menu = menubar.addMenu("Options")
        self._add_menu_button(options_menu, "Spectrogram", "Configure spectrogram display settings",
                             lambda: self.initialize_frame('Spectrogram'))

    def _add_menu_button(self, menu, text, tooltip, callback):
        """Helper method to create a button-like menu item with tooltip"""
        action = QWidgetAction(menu)
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.setCursor(Qt.PointingHandCursor)

        # ONLY CHANGE THE FONT SIZE - keep other styling original
        font = button.font()
        font.setPointSize(15)  # Adjust this value as needed (default is usually 9-10)
        button.setFont(font)

        button.setStyleSheet("""
            QToolButton {
                background-color: transparent;
                color: white;
                border: none;
                padding: 0.75em 1.5em;
                text-align: left;
                min-width: 12em;
                min-height: 2.25em;
                font-size: 16pt;    #hover message font size
            }
            QToolButton:hover {
                background-color: #3498db;
                border-radius: 0.2em;
            }
        """)
        button.clicked.connect(callback)
        action.setDefaultWidget(button)
        menu.addAction(action)
        return action

    def show_separator_tool(self):
        """Show the separator tool in a new window"""
        if not hasattr(self, 'separator_window') or not self.separator_window.isVisible():
            self.separator_window = FundamentalHarmonicsSeparator()
            self.separator_window.show()
            
            # If you have audio loaded in the main window, pass it to the separator
            if hasattr(self, 'current_audio') and self.current_audio is not None:
                self.separator_window.load_signal(self.current_audio, self.current_fs)

    def launch_tuner(self):
        """Launch the live audio tuner"""
        try:
            # First check if we already have a tuner running
            if hasattr(self, 'tuner_window') and self.tuner_window.isVisible():
                self.tuner_window.raise_()
                return
            
            # Import the tuner module if it's in a separate file
            from your_tuner_module import TunerWindow
            
            # Create and show the tuner window
            self.tuner_window = TunerWindow(parent=self)
            self.tuner_window.show()
            
            # Connect close event for cleanup
            self.tuner_window.destroyed.connect(self.cleanup_tuner)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not launch tuner: {str(e)}")

    def cleanup_tuner(self):
        """Clean up tuner resources"""
        if hasattr(self, 'tuner_window'):
            try:
                self.tuner_window.close()
                self.tuner_window.deleteLater()
            except:
                pass
            del self.tuner_window


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