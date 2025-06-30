import time
import threading
import pyaudio
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write
from pathlib import Path
from PyQt5.QtWidgets import (QSpinBox, QApplication, QWidget, QDialog, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont  # Added import
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector, Button
from controlMenu import ControlMenu


class Record(QWidget):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.master = master
        self.isrecording = False
        self.fs = 44100
        self.selectedAudio = np.empty(1)
        self.recording_start_time = 0
        self.frames = []
        
        # Track control windows created from recordings
        self.control_windows = []
        
        self.setupUI()
        
    def setupUI(self):
        self.setWindowTitle("Audio Recorder")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        # Main layout with minimal margins
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Top control row - everything in one line
        control_row = QHBoxLayout()
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(10)
        
        # Record button (red)
        self.record_button = QPushButton("⏺ Record")
        self.record_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:disabled {
                background-color: #aa3333;
            }
        """)
        self.record_button.clicked.connect(self.start_recording)
        
        # Stop button (neutral color)
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #666666;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #888888;
            }
            QPushButton:disabled {
                background-color: #444444;
            }
        """)
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        
        # Time display
        self.time_label = QLabel("00:00")
        self.time_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("min-width: 80px;")
        
        # Max record time dropdown
        time_limit_layout = QHBoxLayout()
        time_limit_layout.setSpacing(5)
        time_limit_label = QLabel("Max:")
        time_limit_label.setFont(QFont("Arial", 10))
        self.time_spinbox = QSpinBox()
        self.time_spinbox.setRange(1, 600)
        self.time_spinbox.setValue(30)
        self.time_spinbox.setSuffix("s")
        self.time_spinbox.setFixedWidth(80)
        
        # Enlarged Help Button
        self.help_button = QPushButton("🛈 Help")
        self.help_button.setFont(QFont("Arial", 18))  # Increased from 10 to 12
        self.help_button.setFixedWidth(120)  # Increased from 70 to 90
        self.help_button.setFixedHeight(35)  # Added fixed height
        self.help_button.clicked.connect(lambda: self.controller.help.createHelpMenu(7))
        self.help_button.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #777777;
            }
        """)
        
        # Enlarged Load Button
        self.load_button = QPushButton("Load to Controller")
        self.load_button.setFont(QFont("Arial", 15))  # Increased from 10 to 12
        self.load_button.setFixedHeight(35)  # Added fixed height
        self.load_button.setVisible(False)
        self.load_button.clicked.connect(self.load_to_controller)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #4477ff;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #6699ff;
            }
        """)
        
        # Add widgets to control row
        control_row.addWidget(self.record_button)
        control_row.addWidget(self.stop_button)
        control_row.addWidget(self.time_label)
        control_row.addWidget(time_limit_label)
        control_row.addWidget(self.time_spinbox)
        control_row.addWidget(self.help_button)
        control_row.addWidget(self.load_button)
        control_row.addStretch()
        
        # Timer setup
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time_display)
        self.auto_stop_timer = QTimer(self)
        self.auto_stop_timer.setSingleShot(True)
        self.auto_stop_timer.timeout.connect(self.stop_recording)
        
        # Plot area
        self.fig = Figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.setVisible(False)
        self.toolbar.setVisible(False)
        
        # Add everything to main layout
        main_layout.addLayout(control_row)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        
        self.setLayout(main_layout)

        
    def start_recording(self):
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.isrecording = True
        self.frames = []
        self.recording_start_time = time.time()
        
        self.timer.start(200)
        self.max_record_time = self.time_spinbox.value()
        self.auto_stop_timer.start(self.max_record_time * 1000)
                
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()
        
    def stop_recording(self):
        if not self.isrecording:
            return
        
        self.isrecording = False
        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.timer.stop()
        self.auto_stop_timer.stop()
        
        self.recording_thread.join()
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
        rec_int = np.frombuffer(b"".join(self.frames), dtype=np.int16)
        duration = len(rec_int) / self.fs
        time_axis = np.linspace(0, duration, len(rec_int))
        
        # Save and reload as float
        recording_dir = Path("wav")
        recording_dir.mkdir(exist_ok=True)
        write(recording_dir / "recording.wav", self.fs, rec_int)
        rec_float, _ = sf.read(recording_dir / "recording.wav", dtype='float32')
        
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
        
        self.setup_span_selector(time_axis, rec_float)
        self.canvas.setVisible(True)
        self.toolbar.setVisible(True)
        self.canvas.draw()
        self.load_button.setVisible(True)

    def load_to_controller(self):
        """Load recorded audio (or selection) into a new ControlMenu window."""
        try:
            # Determine what audio to load
            if hasattr(self, 'selectedAudio') and len(self.selectedAudio) > 1:
                audio_to_load = self.selectedAudio
            else:
                audio_to_load, _ = sf.read("wav/recording.wav", dtype='float32')
            
            duration = len(audio_to_load) / self.fs
            title = f"Recording {time.strftime('%Y-%m-%d %H:%M')}"

            # Create ControlMenu window
            control_window = ControlMenu(title, self.fs, audio_to_load, duration, self.controller)
            self.control_windows.append(control_window)

            # Clean up when window is closed
            def handle_close():
                if control_window in self.control_windows:
                    self.control_windows.remove(control_window)
                if hasattr(self.controller, 'update_windows_menu'):
                    self.controller.update_windows_menu()
            
            control_window.destroyed.connect(handle_close)
            
            # Update windows menu
            if hasattr(self.controller, 'update_windows_menu'):
                self.controller.update_windows_menu()

            control_window.show()
            control_window.activateWindow()

        except Exception as e:
            print(f"Error loading to controller: {e}")
            QMessageBox.critical(self, "Error", f"Could not load to controller:\n{str(e)}")

            
    def setup_span_selector(self, time_axis, audio):
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
