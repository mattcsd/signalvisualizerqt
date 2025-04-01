import sys
import colorednoise as cn
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QSlider, QLineEdit, QPushButton, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector, Button

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
        
        # Noise type selection
        type_layout = QHBoxLayout()
        type_label = QLabel('Noise type:')
        self.type_combo = QComboBox()
        self.type_combo.addItems(['White noise', 'Pink noise', 'Brown noise'])
        self.type_combo.setCurrentText(self.noise_type)
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_combo)
        self.layout.addLayout(type_layout)
        
        # Amplitude control
        ampl_layout = QHBoxLayout()
        ampl_label = QLabel('Max. amplitude:')
        self.ampl_slider = QSlider(Qt.Horizontal)
        self.ampl_slider.setRange(0, 100)
        self.ampl_slider.setValue(int(self.amplitude * 100))
        self.ampl_entry = QLineEdit(f"{self.amplitude:.2f}")
        self.ampl_entry.setFixedWidth(80)
        ampl_layout.addWidget(ampl_label)
        ampl_layout.addWidget(self.ampl_slider)
        ampl_layout.addWidget(self.ampl_entry)
        self.layout.addLayout(ampl_layout)
        
        # Duration control
        dura_layout = QHBoxLayout()
        dura_label = QLabel('Total duration (s):')
        self.dura_slider = QSlider(Qt.Horizontal)
        self.dura_slider.setRange(1, 3000)  # 0.01 to 30.00 in steps of 0.01
        self.dura_slider.setValue(int(self.duration * 100))
        self.dura_entry = QLineEdit(f"{self.duration:.2f}")
        self.dura_entry.setFixedWidth(80)
        dura_layout.addWidget(dura_label)
        dura_layout.addWidget(self.dura_slider)
        dura_layout.addWidget(self.dura_entry)
        self.layout.addLayout(dura_layout)
        
        # Sample rate control
        fs_layout = QHBoxLayout()
        fs_label = QLabel('Fs (Hz):')
        self.fs_entry = QLineEdit(str(self.fs))
        self.fs_entry.setFixedWidth(80)
        fs_layout.addWidget(fs_label)
        fs_layout.addWidget(self.fs_entry)
        self.layout.addLayout(fs_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.plot_button = QPushButton('Plot')
        self.save_button = QPushButton('Save')
        self.help_button = QPushButton('🛈')
        self.help_button.setFixedWidth(30)
        
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()
        button_layout.addWidget(self.help_button)
        button_layout.addWidget(self.plot_button)
        self.layout.addLayout(button_layout)
        
        # Connect signals
        self.ampl_slider.valueChanged.connect(self.update_amplitude)
        self.dura_slider.valueChanged.connect(self.update_duration)
        self.ampl_entry.editingFinished.connect(self.update_amplitude_from_entry)
        self.dura_entry.editingFinished.connect(self.update_duration_from_entry)
        self.plot_button.clicked.connect(self.plot_noise)
        self.save_button.clicked.connect(self.save_default_values)
        self.help_button.clicked.connect(lambda: self.controller.help.createHelpMenu(5))
        
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
            ['PURE TONE','\t duration', default_values[1][2],'\t amplitude', default_values[1][4],'\t fs', default_values[1][6],'\t offset', default_values[1][8],'\t frequency', default_values[1][10],'\t phase',  default_values[1][12]],
            ['SQUARE WAVE','\t duration', default_values[2][2],'\t amplitude', default_values[2][4],'\t fs', default_values[2][6],'\t offset', default_values[2][8],'\t frequency', default_values[2][10],'\t phase', default_values[2][12],'\t active cycle', default_values[2][14]],
            ['SAWTOOTH WAVE','\t duration', default_values[3][2],'\t amplitude', default_values[3][4],'\t fs', default_values[3][6],'\t offset', default_values[3][8],'\t frequency', default_values[3][10],'\t phase', default_values[3][12],'\t max position', default_values[3][14]],
            ['FREE ADD OF PT','\t duration', default_values[4][2],'\t octave', default_values[4][4],'\t freq1', default_values[4][6],'\t freq2', default_values[4][8],'\t freq3', default_values[4][10],'\t freq4', default_values[4][12],'\t freq5', default_values[4][14],'\t freq6', default_values[4][16],'\t amp1', default_values[4][18],'\t amp2', default_values[4][20],'\t amp3', default_values[4][22],'\t amp4', default_values[4][24],'\t amp5', default_values[4][26],'\t amp6', default_values[4][28]],
            ['SPECTROGRAM','\t colormap', default_values[5][2]]
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
        
        if choice == 'White noise':
            beta = 0
        elif choice == 'Pink noise':
            beta = 1
        elif choice == 'Brown noise':
            beta = 2
            
        time = np.linspace(start=0, stop=self.duration, num=samples, endpoint=False)
        noiseGaussian = cn.powerlaw_psd_gaussian(beta, samples)
        noise = self.amplitude * noiseGaussian / max(abs(noiseGaussian))
        
        # Create plot window
        self.plot_window = PlotWindow(choice, self.fs, time, noise, self.duration, self.controller)
        self.plot_window.show()
        
    def create_control_menu(self, name, fs, audio, duration):
        """Helper method to create ControlMenu with required parameters"""
        return ControlMenu(name, fs, audio, duration, self.controller, self)


class PlotWindow(QMainWindow):
    def __init__(self, title, fs, time, audio, duration, controller):
        super().__init__()
        self.controller = controller
        self.fs = fs
        self.time = time
        self.audio = audio
        self.duration = duration
        self.title = title
        self.selectedAudio = np.empty(1)
        self.setWindowTitle(title)
        self.setup_ui()
        
    def setup_ui(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.ax = self.figure.add_subplot(111)
        
        # Add load button
        self.load_ax = self.figure.add_axes([0.8, 0.01, 0.09, 0.05])
        self.load_button = Button(self.load_ax, 'Load')
        self.load_button.on_clicked(self.load_selection)
        
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        self.plot()
        
    def plot(self):
        self.ax.clear()
        self.ax.plot(self.time, self.audio)
        self.ax.set(xlim=[0, self.duration], xlabel='Time (s)', ylabel='Amplitude')
        self.ax.axhline(y=0, color='black', linewidth='0.5', linestyle='--')
        
        # Add span selector
        self.span = SpanSelector(self.ax, self.listen_fragment, 'horizontal', 
                               useblit=True, interactive=True, drag_from_anywhere=True)
        
        self.canvas.draw()
        
    def listen_fragment(self, xmin, xmax):
        ini, end = np.searchsorted(self.time, (xmin, xmax))
        self.selectedAudio = self.audio[ini:end+1]
        sd.play(self.selectedAudio, self.fs)
        
    def load_selection(self, event):
        try:
            if self.selectedAudio.shape == (1,): 
                # Create control menu with full audio
                control_menu = ControlMenu(
                    fileName=self.title,
                    fs=self.fs,
                    audioFrag=self.audio,
                    duration=self.duration,
                    controller=self.controller,
                    parent=self
                )
            else:
                # Create control menu with selected fragment
                time = np.arange(0, len(self.selectedAudio)/self.fs, 1/self.fs)
                durSelec = max(time)
                control_menu = ControlMenu(
                    fileName=self.title,
                    fs=self.fs,
                    audioFrag=self.selectedAudio,
                    duration=durSelec,
                    controller=self.controller,
                    parent=self
                )
            control_menu.show()
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create control menu: {str(e)}")

