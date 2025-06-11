import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import butter, lfilter
import sounddevice as sd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QMessageBox, 
                            QDoubleSpinBox, QFormLayout)
from PyQt5.QtCore import Qt

class FundamentalHarmonicsSeparator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_signal = None
        self.fs = 44100
        self.f0 = None
        self.current_stream = None
        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        
        # Control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Fundamental frequency controls
        freq_panel = QWidget()
        freq_layout = QFormLayout(freq_panel)
        
        self.f0_spinbox = QDoubleSpinBox()
        self.f0_spinbox.setRange(20, 2000)
        self.f0_spinbox.setValue(440)
        self.f0_spinbox.setSingleStep(1)
        self.f0_spinbox.valueChanged.connect(self.update_fundamental)
        
        self.bandwidth_spinbox = QDoubleSpinBox()
        self.bandwidth_spinbox.setRange(1, 100)
        self.bandwidth_spinbox.setValue(20)
        self.bandwidth_spinbox.setSingleStep(1)
        self.bandwidth_spinbox.valueChanged.connect(self.update_fundamental)
        
        freq_layout.addRow("Fundamental (Hz):", self.f0_spinbox)
        freq_layout.addRow("Bandwidth (Hz):", self.bandwidth_spinbox)
        
        # Buttons
        self.btn_estimate = QPushButton("Estimate F0")
        self.btn_estimate.clicked.connect(self.estimate_fundamental)
        
        self.btn_play_original = QPushButton("Original")
        self.btn_play_original.clicked.connect(lambda: self.play_audio(self.original_signal))
        
        self.btn_play_fundamental = QPushButton("Fundamental")
        self.btn_play_fundamental.clicked.connect(lambda: self.play_audio(self.fundamental))
        
        self.btn_play_harmonics = QPushButton("Harmonics")
        self.btn_play_harmonics.clicked.connect(lambda: self.play_audio(self.harmonics))
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_audio)
        
        # Add to control layout
        control_layout.addWidget(freq_panel)
        control_layout.addWidget(self.btn_estimate)
        control_layout.addWidget(self.btn_play_original)
        control_layout.addWidget(self.btn_play_fundamental)
        control_layout.addWidget(self.btn_play_harmonics)
        control_layout.addWidget(self.btn_stop)
        
        # Plot area
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        
        # Add to main layout
        self.layout.addWidget(control_panel)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def load_signal(self, signal, fs):
        """Load a signal for processing"""
        self.original_signal = signal
        self.fs = fs
        self.plot_signals()

    def estimate_fundamental(self):
        """Estimate fundamental frequency using autocorrelation"""
        if self.original_signal is None:
            QMessageBox.warning(self, "Warning", "No signal loaded!")
            return
            
        signal = self.original_signal - np.mean(self.original_signal)
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find first peak after zero-lag
        peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & 
                        (autocorr[1:-1] > autocorr[2:]))[0] + 1
        
        if len(peaks) > 0:
            self.f0 = self.fs / peaks[0]
            self.f0_spinbox.setValue(self.f0)
            self.separate_components()
            self.plot_signals()
        else:
            QMessageBox.warning(self, "Warning", "Could not estimate fundamental frequency!")

    def update_fundamental(self):
        """Update separation when parameters change"""
        self.f0 = self.f0_spinbox.value()
        if self.original_signal is not None:
            self.separate_components()
            self.plot_signals()

    def separate_components(self):
        """Separate signal into fundamental and harmonics"""
        if self.f0 is None or self.original_signal is None:
            return
            
        bandwidth = self.bandwidth_spinbox.value()
        
        # Bandpass filter for fundamental
        nyq = 0.5 * self.fs
        low = (self.f0 - bandwidth/2) / nyq
        high = (self.f0 + bandwidth/2) / nyq
        b, a = butter(4, [low, high], btype='band')
        self.fundamental = lfilter(b, a, self.original_signal)
        
        # Notch filter for harmonics
        b, a = butter(4, [low, high], btype='bandstop')
        self.harmonics = lfilter(b, a, self.original_signal)

    def plot_signals(self):
        """Plot original, fundamental, and harmonics"""
        self.figure.clear()
        
        if self.original_signal is None:
            return
            
        t = np.linspace(0, len(self.original_signal)/self.fs, len(self.original_signal))
        
        # Plot original
        ax1 = self.figure.add_subplot(311)
        ax1.plot(t, self.original_signal)
        ax1.set_title("Original Signal")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        
        # Plot fundamental if available
        ax2 = self.figure.add_subplot(312)
        if hasattr(self, 'fundamental'):
            ax2.plot(t, self.fundamental, 'g')
            ax2.set_title(f"Fundamental Component ({self.f0:.1f} Hz)")
        else:
            ax2.set_title("Fundamental Component (Not calculated)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, alpha=0.3)
        
        # Plot harmonics if available
        ax3 = self.figure.add_subplot(313)
        if hasattr(self, 'harmonics'):
            ax3.plot(t, self.harmonics, 'r')
            ax3.set_title("Harmonics Component")
        else:
            ax3.set_title("Harmonics Component (Not calculated)")
        ax3.set_ylabel("Amplitude")
        ax3.set_xlabel("Time (s)")
        ax3.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def play_audio(self, signal):
        """Play audio signal"""
        if signal is None:
            return
        self.stop_audio()
        self.current_stream = sd.play(signal, self.fs)

    def stop_audio(self):
        """Stop audio playback"""
        if self.current_stream is not None:
            sd.stop()
            self.current_stream = None

    def cleanup(self):
        """Clean up resources"""
        self.stop_audio()
        if hasattr(self, 'current_stream') and self.current_stream is not None:
            self.current_stream.close()