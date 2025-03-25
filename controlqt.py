import sys
import math
import librosa
import librosa.display
import parselmouth
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor, SpanSelector, MultiCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QRadioButton, QCheckBox, QMessageBox, QGridLayout, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5 import QtGui  # Import QtGui for QDoubleValidator
from scipy import signal
from scipy.io.wavfile import write


class ControlMenu(QMainWindow):
    def __init__(self, fileName, fs, audioFrag, duration, controller):
        super().__init__()
        self.fileName = fileName
        self.audio = audioFrag  # Audio array of the fragment
        self.fs = fs  # Sample frequency of the audio (Hz)
        self.time = np.arange(0, len(self.audio) / self.fs)  # Time array of the audio
        self.duration = duration  # Duration of the audio (s)
        self.lenAudio = len(self.audio)  # Length of the audio array
        self.controller = controller
        np.seterr(divide='ignore')  # Turn off "RuntimeWarning: divide by zero encountered in log10"

        # Ensure 'self.time' and 'self.audio' have the same first dimension
        if len(self.time) < len(self.audio):
            self.audio = self.audio[:-1].copy()
        elif len(self.time) > len(self.audio):
            self.time = self.time[:-1].copy()

        # Initialize UI
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.fileName)
        self.setGeometry(100, 100, 750, 575)

        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QGridLayout(main_widget)

        # Labels
        labels = {
            'Choose an option': (0, 0, 1, 2),
            'Window': (1, 0),
            'nfft': (3, 0),
            'Method': (11, 0),
            'Filter type': (2, 2),
            'Window length (s)': (2, 0),
            'Overlap (s)': (4, 0),
            'Min frequency (Hz)': (6, 0),
            'Max frequency (Hz)': (7, 0),
            'Pitch floor (Hz)': (12, 0),
            'Pitch ceiling (Hz)': (13, 0),
            'Fund. freq. multiplication': (3, 2),
            'First harmonic frequency': (4, 2),
            'Percentage (%)': (5, 2),
            'Fcut': (6, 2),
            'Fcut1': (7, 2),
            'Fcut2': (8, 2),
            'Beta': (10, 2),
            'Fs: ' + str(self.fs) + ' Hz': (13, 3),
            'Spectrogram': (5, 1),
            'Pitch': (10, 1),
            'Filtering': (1, 3),
            'Short-Time-Energy': (9, 3),
            'Drawing style': (8, 0)
        }

        for text, pos in labels.items():
            label = QLabel(text)
            if text == 'Choose an option' or text.startswith('Fs:'):
                label.setStyleSheet("font-weight: bold;")
            layout.addWidget(label, *pos)

        # Entries
        self.entries = {
            'size': QLineEdit(str(0.03)),
            'over': QLineEdit(str(0.01)),
            'minf': QLineEdit(),
            'maxf': QLineEdit(str(self.fs / 2)),
            'minp': QLineEdit(str(75.0)),
            'maxp': QLineEdit(str(600.0)),
            'fund': QLineEdit(str(1)),
            'cent': QLineEdit(str(400)),
            'perc': QLineEdit(str(10.0)),
            'fcut': QLineEdit(str(1000)),
            'cut1': QLineEdit(str(200)),
            'cut2': QLineEdit(str(600)),
            'beta': QLineEdit()
        }

        for key, entry in self.entries.items():
            entry.setValidator(self.float_validator())
            if key in ['minp', 'maxp', 'fund', 'cent', 'perc', 'fcut', 'cut1', 'cut2', 'beta']:
                entry.setDisabled(True)

        layout.addWidget(self.entries['size'], 2, 1)
        layout.addWidget(self.entries['over'], 4, 1)
        layout.addWidget(self.entries['minf'], 6, 1)
        layout.addWidget(self.entries['maxf'], 7, 1)
        layout.addWidget(self.entries['minp'], 12, 1)
        layout.addWidget(self.entries['maxp'], 13, 1)
        layout.addWidget(self.entries['fund'], 3, 3)
        layout.addWidget(self.entries['cent'], 4, 3)
        layout.addWidget(self.entries['perc'], 5, 3)
        layout.addWidget(self.entries['fcut'], 6, 3)
        layout.addWidget(self.entries['cut1'], 7, 3)
        layout.addWidget(self.entries['cut2'], 8, 3)
        layout.addWidget(self.entries['beta'], 10, 3)

        # Radio Buttons
        self.rdb_lin = QRadioButton('linear')
        self.rdb_mel = QRadioButton('mel')
        self.rdb_lin.setChecked(True)
        layout.addWidget(self.rdb_lin, 8, 1)
        layout.addWidget(self.rdb_mel, 8, 1, 1, 2)

        # Checkbox
        self.chk_pitch = QCheckBox('Show pitch')
        self.chk_pitch.stateChanged.connect(self.pitch_checkbox_changed)
        layout.addWidget(self.chk_pitch, 9, 1)

        # Buttons
        self.btn_plot = QPushButton('Plot')
        self.btn_plot.clicked.connect(self.plot_figure)
        layout.addWidget(self.btn_plot, 14, 3)

        # Option Menus
        self.options = ['FT', 'STFT', 'Spectrogram', 'STFT + Spect', 'Short-Time-Energy', 'Pitch', 'Spectral Centroid', 'Filtering']
        self.opt_wind = ['Bartlett', 'Blackman', 'Hamming', 'Hanning', 'Kaiser']
        self.opt_nfft = [2 ** i for i in range(9, 20)]
        self.opt_meth = ['Autocorrelation', 'Cross-correlation', 'Subharmonics', 'Spinet']
        self.opt_pass = ['Harmonic', 'Lowpass', 'Highpass', 'Bandpass', 'Bandstop']

        self.dd_opts = QComboBox()
        self.dd_opts.addItems(self.options)
        self.dd_opts.setCurrentText('Spectrogram')
        self.dd_opts.currentTextChanged.connect(self.display_options)
        layout.addWidget(self.dd_opts, 0, 2)

        self.dd_wind = QComboBox()
        self.dd_wind.addItems(self.opt_wind)
        layout.addWidget(self.dd_wind, 1, 1)

        self.dd_nfft = QComboBox()
        self.dd_nfft.addItems(map(str, self.opt_nfft))
        layout.addWidget(self.dd_nfft, 3, 1)

        self.dd_meth = QComboBox()
        self.dd_meth.addItems(self.opt_meth)
        self.dd_meth.setDisabled(True)
        layout.addWidget(self.dd_meth, 11, 1)

        self.dd_pass = QComboBox()
        self.dd_pass.addItems(self.opt_pass)
        self.dd_pass.setDisabled(True)
        self.dd_pass.currentTextChanged.connect(self.display_filter_options)
        layout.addWidget(self.dd_pass, 2, 3)

    def float_validator(self):
        return QtGui.QDoubleValidator()  # Use QDoubleValidator from QtGui

    def pitch_checkbox_changed(self):
        show_pitch = self.chk_pitch.isChecked()
        self.dd_meth.setEnabled(show_pitch)
        self.entries['minp'].setEnabled(show_pitch)
        self.entries['maxp'].setEnabled(show_pitch)

    def display_options(self, choice):
        self.entries['over'].setDisabled(choice in ['FT', 'STFT', 'Pitch', 'Filtering'])
        self.entries['size'].setDisabled(choice in ['FT', 'Pitch', 'Filtering'])
        self.dd_pass.setEnabled(choice == 'Filtering')
        self.rdb_lin.setEnabled(choice in ['Spectrogram', 'STFT + Spect', 'Spectral Centroid', 'Filtering'])
        self.rdb_mel.setEnabled(choice in ['Spectrogram', 'STFT + Spect', 'Spectral Centroid', 'Filtering'])
        self.entries['minf'].setEnabled(choice in ['Spectrogram', 'STFT + Spect', 'Spectral Centroid'])
        self.entries['maxf'].setEnabled(choice in ['Spectrogram', 'STFT + Spect', 'Spectral Centroid'])
        self.dd_wind.setEnabled(choice in ['STFT', 'Spectrogram', 'STFT + Spect', 'Spectral Centroid', 'Short-Time-Energy'])
        self.dd_nfft.setEnabled(choice in ['STFT', 'Spectrogram', 'STFT + Spect', 'Spectral Centroid'])
        self.entries['beta'].setEnabled(choice == 'Short-Time-Energy')
        self.chk_pitch.setEnabled(choice == 'Spectrogram')

    def display_filter_options(self, choice):
        self.entries['fcut'].setEnabled(choice in ['Lowpass', 'Highpass'])
        self.entries['cut1'].setEnabled(choice in ['Bandpass', 'Bandstop'])
        self.entries['cut2'].setEnabled(choice in ['Bandpass', 'Bandstop'])
        self.entries['fund'].setEnabled(choice == 'Harmonic')
        self.entries['cent'].setEnabled(choice == 'Harmonic')

    def plot_figure(self):
        # Collect user inputs and call the appropriate plotting method
        pass  # Implement plotting logic here

    def closeEvent(self, event):
        plt.close('all')
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ControlMenu("example.wav", 44100, np.random.rand(44100), 1.0, None)
    window.show()
    sys.exit(app.exec_())