# beat_visualizer_live.py
import numpy as np
import soundfile as sf
import sounddevice as sd
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer


class BeatFrequencyVisualizer(QWidget):
    def __init__(self, parent=None, controller=None):
        super().__init__(parent)
        self.controller = controller
        self.fs = 44100  # Will be updated after loading
        self.audio_data = None
        self.ptr = 0
        self.chunk_size = 1024

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_fft)

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.label = QLabel("Real-Time STFT from Lute Recording")
        layout.addWidget(self.label)

        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.ax = self.canvas.figure.add_subplot(111)
        self.line, = self.ax.plot([], [])
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, self.fs // 2)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude")
        layout.addWidget(self.canvas)

        self.play_button = QPushButton("Play & Analyze")
        self.play_button.clicked.connect(self.play_audio)
        layout.addWidget(self.play_button)

    def play_audio(self):
        # Load audio if not loaded
        if self.audio_data is None:
            path = "recordings/lute_beat_demo.wav"
            self.audio_data, self.fs = sf.read(path)
            if self.audio_data.ndim > 1:
                self.audio_data = self.audio_data[:, 0]  # Use mono

        self.ptr = 0  # Reset pointer

        # Start audio stream
        self.stream = sd.OutputStream(samplerate=self.fs, channels=1, callback=self.audio_callback)
        self.stream.start()

        # Start timer for FFT update
        self.timer.start(int(1000 * self.chunk_size / self.fs))  # update based on chunk duration

    def audio_callback(self, outdata, frames, time, status):
        if self.ptr + frames > len(self.audio_data):
            outdata[:] = np.zeros((frames, 1))
            self.stream.stop()
            self.timer.stop()
            return

        chunk = self.audio_data[self.ptr:self.ptr+frames]
        outdata[:len(chunk), 0] = chunk
        self.ptr += frames

    def update_fft(self):
        if self.ptr < self.chunk_size:
            return
        window = self.audio_data[self.ptr - self.chunk_size:self.ptr]
        fft = np.abs(np.fft.rfft(window * np.hanning(len(window))))
        freqs = np.fft.rfftfreq(len(window), 1 / self.fs)

        self.line.set_data(freqs, fft / np.max(fft))  # Normalize
        self.ax.set_xlim(0, 2000)  # Focus on low freqs
        self.ax.set_ylim(0, 1)
        self.canvas.draw_idle()
