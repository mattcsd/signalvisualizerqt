import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider, QLineEdit, QPushButton,
    QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import sounddevice as sd
import colorednoise as cn  # Ensure this is installed
from auxiliar import Auxiliar
from controlMenu import ControlMenu

class Noise(QWidget):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.master = master
        self.aux = Auxiliar()
        self.selectedAudio = np.empty(1)
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle('Generate noise')
        self.layout = QVBoxLayout(self)

        # Read default values
        default_values = self.aux.readFromCsv()
        self.duration = float(default_values[0][2])
        self.amplitude = float(default_values[0][4])
        self.fs = int(default_values[0][6])
        self.noise_type = default_values[0][8]

        # Noise type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel('Noise type:'))
        self.type_combo = QComboBox()
        self.type_combo.addItems(['White noise', 'Pink noise', 'Brown noise'])
        self.type_combo.setCurrentText(self.noise_type)
        type_layout.addWidget(self.type_combo)
        self.layout.addLayout(type_layout)

        # Amplitude
        ampl_layout = QHBoxLayout()
        ampl_layout.addWidget(QLabel('Max. amplitude:'))
        self.ampl_slider = QSlider(Qt.Horizontal)
        self.ampl_slider.setRange(0, 100)
        self.ampl_slider.setValue(int(self.amplitude * 100))
        self.ampl_entry = QLineEdit(f"{self.amplitude:.2f}")
        self.ampl_entry.setFixedWidth(80)
        ampl_layout.addWidget(self.ampl_slider)
        ampl_layout.addWidget(self.ampl_entry)
        self.layout.addLayout(ampl_layout)

        # Duration
        dura_layout = QHBoxLayout()
        dura_layout.addWidget(QLabel('Total duration (s):'))
        self.dura_slider = QSlider(Qt.Horizontal)
        self.dura_slider.setRange(1, 3000)
        self.dura_slider.setValue(int(self.duration * 100))
        self.dura_entry = QLineEdit(f"{self.duration:.2f}")
        self.dura_entry.setFixedWidth(80)
        dura_layout.addWidget(self.dura_slider)
        dura_layout.addWidget(self.dura_entry)
        self.layout.addLayout(dura_layout)

        # Sample rate
        fs_layout = QHBoxLayout()
        fs_layout.addWidget(QLabel('Fs (Hz):'))
        self.fs_entry = QLineEdit(str(self.fs))
        self.fs_entry.setFixedWidth(80)
        fs_layout.addWidget(self.fs_entry)
        self.layout.addLayout(fs_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.plot_button = QPushButton('Plot')
        self.controller_button = QPushButton('Load to Controller')
        self.save_button = QPushButton('Save')
        self.help_button = QPushButton('🛈')
        self.help_button.setFixedWidth(30)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()
        button_layout.addWidget(self.controller_button)
        button_layout.addWidget(self.help_button)
        button_layout.addWidget(self.plot_button)
        self.layout.addLayout(button_layout)

        # Plot area
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.figure.add_subplot(111)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        # Connect
        self.ampl_slider.valueChanged.connect(self.update_amplitude)
        self.dura_slider.valueChanged.connect(self.update_duration)
        self.ampl_entry.editingFinished.connect(self.update_amplitude_from_entry)
        self.dura_entry.editingFinished.connect(self.update_duration_from_entry)
        self.plot_button.clicked.connect(self.plot_noise)
        self.controller_button.clicked.connect(self.load_to_controller)
        self.save_button.clicked.connect(self.save_default_values)
        self.help_button.clicked.connect(lambda: self.controller.help.createHelpMenu(5))

        # Initial plot
        self.plot_noise()

    def load_to_controller(self):
        """Load the generated noise to a new controller window"""
        try:
            # Re-plot and ensure audio is up to date
            self.plot_noise()
            
            # Use entire audio if no fragment was selected
            audio_to_load = (
                self.selectedAudio if self.selectedAudio.size > 1 else self.audio
            )
            duration = self.duration
            fs = self.fs
            
            # Create window title
            noise_type = self.type_combo.currentText()
            title = f"{noise_type} ({duration:.2f}s)"
            
            # Ensure controller has required interface
            if not hasattr(self.controller, 'adse'):
                from PyQt5.QtWidgets import QWidget
                self.controller = QWidget()
                self.controller.adse = type('', (), {})()
                self.controller.adse.advancedSettings = lambda: print("Advanced settings not available")
            
            # Create controller window
            control_window = ControlMenu(title, fs, audio_to_load, duration, self.controller)
            
            # Track windows
            if not hasattr(self, 'control_windows'):
                self.control_windows = []
            self.control_windows.append(control_window)
            
            # Cleanup when closed
            control_window.destroyed.connect(
                lambda: self.control_windows.remove(control_window) 
                if control_window in self.control_windows else None
            )
            
            control_window.show()
            control_window.activateWindow()
            
        except Exception as e:
            print(f"Error loading noise to controller: {e}")
            QMessageBox.critical(self, "Error", f"Could not load noise to controller: {str(e)}")


    def update_amplitude(self, value):
        self.amplitude = value / 100
        self.ampl_entry.setText(f"{self.amplitude:.2f}")

    def update_duration(self, value):
        self.duration = value / 100
        self.dura_entry.setText(f"{self.duration:.2f}")

    def update_amplitude_from_entry(self):
        try:
            value = float(self.ampl_entry.text())
            if 0 <= value <= 1:
                self.amplitude = value
                self.ampl_slider.setValue(int(value * 100))
            else:
                self.ampl_entry.setText(f"{self.amplitude:.2f}")
        except ValueError:
            self.ampl_entry.setText(f"{self.amplitude:.2f}")

    def update_duration_from_entry(self):
        try:
            value = float(self.dura_entry.text())
            if 0.01 <= value <= 30:
                self.duration = value
                self.dura_slider.setValue(int(value * 100))
            else:
                self.dura_entry.setText(f"{self.duration:.2f}")
        except ValueError:
            self.dura_entry.setText(f"{self.duration:.2f}")

    def save_default_values(self):
        default_values = self.aux.readFromCsv()
        choice = self.type_combo.currentText()
        amplitude = self.amplitude
        duration = self.duration
        new_list = [
            ['NOISE','\t duration', duration,'\t amplitude', amplitude,'\t fs', self.fs,'\t noise type', choice],
            *default_values[1:]  # keep rest unchanged
        ]
        self.aux.saveDefaultAsCsv(new_list)

    def plot_noise(self):
        try:
            self.fs = int(self.fs_entry.text())
            if self.fs > 48000:
                self.fs = 48000
                self.fs_entry.setText("48000")
                QMessageBox.warning(self, 'Wrong sample frequency value',
                                    'The sample frequency cannot be greater than 48000 Hz.')
                return
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for sample frequency.')
            return

        choice = self.type_combo.currentText()
        samples = int(self.duration * self.fs)

        beta = {"White noise": 0, "Pink noise": 1, "Brown noise": 2}.get(choice, 1)

        self.time = np.linspace(0, self.duration, samples, endpoint=False)
        noise_raw = cn.powerlaw_psd_gaussian(beta, samples)
        self.audio = self.amplitude * noise_raw / max(abs(noise_raw))

        self.ax.clear()
        self.ax.plot(self.time, self.audio)
        self.ax.set(xlabel='Time (s)', ylabel='Amplitude', xlim=[0, self.duration])
        self.ax.axhline(y=0, color='black', linewidth='0.5', linestyle='--')

        self.span = SpanSelector(
            self.ax, self.listen_fragment, 'horizontal',
            useblit=True, interactive=True, drag_from_anywhere=True
        )

        self.canvas.draw()

    def listen_fragment(self, xmin, xmax):
        ini, end = np.searchsorted(self.time, (xmin, xmax))
        self.selectedAudio = self.audio[ini:end + 1]
        sd.play(self.selectedAudio, self.fs)
