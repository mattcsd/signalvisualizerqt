import time
import threading
import pyaudio
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QDialog, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont  # Added import
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector, Button
from controlMenu import ControlMenu

class Record(QDialog):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.master = master
        self.isrecording = False
        self.fs = 44100
        self.selectedAudio = np.empty(1)
        self.recording_start_time = 0
        self.frames = []
        
        self.setupUI()
        
    def setupUI(self):
        self.setWindowTitle("Audio Recorder")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Recording controls
        control_layout = QHBoxLayout()
        
        self.record_button = QPushButton("⏺")
        self.record_button.setFont(QFont("Arial", 30, QFont.Bold))  # Fixed this line
        self.record_button.clicked.connect(self.start_recording)
        
        self.stop_button = QPushButton("⏹")
        self.stop_button.setFont(QFont("Arial", 30, QFont.Bold))  # Fixed this line
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        
        self.help_button = QPushButton("🛈")
        self.help_button.setFixedWidth(40)
        self.help_button.clicked.connect(lambda: self.controller.help.createHelpMenu(7))
        
        control_layout.addWidget(self.record_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch()
        control_layout.addWidget(self.help_button)
        
        # Timer display
        self.time_label = QLabel("00:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setFont(QFont("", 30))  # Fixed this line
        
        # Timer for updating display
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time_display)
        
        # Figure setup
        self.fig = Figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Initially hide the plot until recording is done
        self.canvas.setVisible(False)
        self.toolbar.setVisible(False)
        
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.time_label)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        
        self.setLayout(main_layout)
        
    def start_recording(self):
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.isrecording = True
        self.frames = []
        self.recording_start_time = time.time()
        self.timer.start(200)  # Update timer every 200ms
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()
        
    def stop_recording(self):
        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.isrecording = False
        self.timer.stop()
        
        # Wait for recording thread to finish
        self.recording_thread.join()
        
        # Process and plot the recording
        self.process_recording()
        
    def record_audio(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.fs,
            input=True,
            frames_per_buffer=1024
        )
        
        while self.isrecording:
            data = stream.read(1024)
            self.frames.append(data)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
    def update_time_display(self):
        duration = time.time() - self.recording_start_time
        mins, secs = divmod(int(duration), 60)
        self.time_label.setText(f"{mins:02d}:{secs:02d}")
        
    def process_recording(self):
        # Convert recorded frames to numpy array
        rec_int = np.frombuffer(b"".join(self.frames), dtype=np.int16)
        duration = len(rec_int) / self.fs
        time_axis = np.linspace(0, duration, len(rec_int))
        
        # Save as WAV file
        recording_dir = Path("wav")
        recording_dir.mkdir(exist_ok=True)
        write(recording_dir / "recording.wav", self.fs, rec_int)
        
        # Read back as float32 for processing
        rec_float, _ = sf.read(recording_dir / "recording.wav", dtype='float32')
        
        # Plot the recording
        self.ax.clear()
        self.ax.plot(time_axis, rec_int)
        self.ax.set(
            xlim=[0, duration],
            xlabel='Time (s)',
            ylabel='Amplitude',
            title='Recording'
        )
        self.ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        self.ax.grid(True, linestyle=':', alpha=0.5)
        
        # Add load button
        self.add_load_button(rec_float, duration)
        
        # Setup audio selection
        self.setup_span_selector(time_axis, rec_float)
        
        # Show the plot
        self.canvas.setVisible(True)
        self.toolbar.setVisible(True)
        self.canvas.draw()
        
    def add_load_button(self, audio, duration):
        # Remove existing button if it exists
        if hasattr(self, 'load_button_ax'):
            self.fig.delaxes(self.load_button_ax)
            
        # Create button axes
        self.load_button_ax = self.fig.add_axes([0.8, 0.01, 0.15, 0.05])
        self.load_button = Button(self.load_button_ax, 'Load to Controller')
        
        def on_load(event):
            if self.selectedAudio.shape == (1,):  # No selection, use entire audio
                audio_to_load = self.ax.lines[0].get_ydata()
                duration_to_load = len(audio_to_load) / self.fs
            else:
                audio_to_load = self.selectedAudio
                duration_to_load = len(audio_to_load) / self.fs
                
            # Create control menu
            cm = ControlMenu("Recording", self.fs, audio_to_load, duration_to_load, self.controller)
            cm.show()
            self.close()
            
        self.load_button.on_clicked(on_load)
        
    def setup_span_selector(self, time_axis, audio):
        # Remove existing span selector if it exists
        if hasattr(self, 'span'):
            self.span.disconnect_events()
            del self.span
            
        def on_select(xmin, xmax):
            if len(audio) <= 1:
                return
                
            idx_min = np.argmax(time_axis >= xmin)
            idx_max = np.argmax(time_axis >= xmax)
            self.selectedAudio = audio[idx_min:idx_max]
            sd.play(self.selectedAudio, self.fs)
            
        self.span = SpanSelector(
            self.ax,
            on_select,
            'horizontal',
            useblit=True,
            interactive=True,
            drag_from_anywhere=True
        )
        
    def show_help(self):
        QMessageBox.information(
            self,
            "Recording Help",
            "Audio Recorder Help\n\n"
            "1. Click the record button (⏺) to start recording\n"
            "2. Click the stop button (⏹) to stop recording\n"
            "3. Select a portion of the recording to play just that section\n"
            "4. Click 'Load to Controller' to send the audio to the control menu\n"
            "   - If no selection is made, the entire recording will be loaded"
        )