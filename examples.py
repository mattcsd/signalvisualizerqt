import numpy as np
import librosa
import librosa.display
import os
import time
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea, 
                            QGroupBox, QPushButton, QMessageBox, 
                            QFormLayout, QSpinBox, QHBoxLayout,
                            QSizePolicy, QApplication)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure

class BeatFrequencyVisualizer(QWidget):
    def __init__(self, parent=None, controller=None):
        super().__init__(parent)
        self.controller = controller
        self.audio_data = None
        self.sample_rate = None
        self.time = None
        self.playback_lines = []
        self.axes = []  # Store references to all axes
        self.backgrounds = None  # Will store the complete figure background
        self.last_update_time = time.time()
        self.update_interval = 0.02  # 20ms for ~50fps

        self.first_playback = True  # Add this flag
        
        # Fixed file path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.fixed_file_path = os.path.join(current_dir, "recordings", "beat.wav")
        
        # Media player setup
        self.media_player = QMediaPlayer()
        self.media_player.setNotifyInterval(20)
        self.media_player.positionChanged.connect(self.update_playback_cursor)
        
        self.init_ui()
        self.load_fixed_audio_file()


    def init_ui(self):
        self.setWindowTitle("Beat Frequency Visualizer")
        self.setMinimumSize(1000, 800)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Analysis parameters - now in a single row
        param_group = QGroupBox("Analysis Parameters")
        param_layout = QHBoxLayout()  # Changed to horizontal layout
        
        self.max_freq_spin = QSpinBox()
        self.max_freq_spin.setRange(100, 10000)
        self.max_freq_spin.setValue(2000)
        
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(256, 4096)
        self.window_size_spin.setValue(1024)
        
        self.hop_size_spin = QSpinBox()
        self.hop_size_spin.setRange(64, 1024)
        self.hop_size_spin.setValue(256)
        
        self.replot_btn = QPushButton("Replot")
        self.replot_btn.clicked.connect(self.plot_spectrogram)
        
        param_layout.addWidget(QLabel("Max Freq (Hz):"))
        param_layout.addWidget(self.max_freq_spin)
        param_layout.addWidget(QLabel("Window Size:"))
        param_layout.addWidget(self.window_size_spin)
        param_layout.addWidget(QLabel("Hop Size:"))
        param_layout.addWidget(self.hop_size_spin)
        param_layout.addWidget(self.replot_btn)
        param_layout.addStretch()
        param_group.setLayout(param_layout)
        
        # Playback controls
        playback_group = QGroupBox("Audio Playback")
        playback_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        
        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.stop_btn)
        playback_group.setLayout(playback_layout)
        
        # Visualization area
        self.figure = Figure(figsize=(10, 7))  # Slightly taller for 3 plots
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # Educational content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.edu_content = QLabel()
        self.edu_content.setWordWrap(True)
        self.edu_content.setTextFormat(Qt.RichText)
        scroll.setWidget(self.edu_content)
        
        # Add widgets to main layout
        main_layout.addWidget(param_group)
        main_layout.addWidget(playback_group)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(scroll, stretch=1)
        
        self.setLayout(main_layout)

    def load_fixed_audio_file(self):
        try:
            if os.path.exists(self.fixed_file_path):
                self.audio_data, self.sample_rate = librosa.load(
                    self.fixed_file_path, 
                    sr=44100,
                    mono=True
                )
                self.time = np.arange(len(self.audio_data)) / self.sample_rate
                
                url = QUrl.fromLocalFile(self.fixed_file_path)
                self.media_player.setMedia(QMediaContent(url))
                
                self.plot_spectrogram()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio: {str(e)}")

    def plot_spectrogram(self):
        if self.audio_data is None:
            return
            
        self.figure.clear()
        self.playback_lines = []
        
        # Create 3 subplots
        gs = self.figure.add_gridspec(3, 1, height_ratios=[1, 1, 2], hspace=0.6)
        
        # 1. Waveform plot
        ax0 = self.figure.add_subplot(gs[0])
        ax0.plot(self.time, self.audio_data, color='b', linewidth=0.5, alpha=0.7)
        self.playback_lines.append(ax0.axvline(x=0, color='r', linewidth=1, animated=True))
        ax0.set_title("Waveform")
        ax0.set_xlim(0, self.time[-1])
        
        # 2. Amplitude envelope plot
        ax1 = self.figure.add_subplot(gs[1], sharex=ax0)
        amplitude = np.abs(self.audio_data)
        smooth_window = int(0.02 * self.sample_rate)
        amplitude_smooth = np.convolve(amplitude, np.ones(smooth_window)/smooth_window, mode='same')
        ax1.plot(self.time, amplitude_smooth, 'b-', linewidth=1)
        self.playback_lines.append(ax1.axvline(x=0, color='r', linewidth=1, animated=True))
        ax1.set_title("Amplitude Envelope")
        ax1.set_ylabel("Amplitude")
        
        # 3. Spectrogram plot
        ax2 = self.figure.add_subplot(gs[2], sharex=ax0)
        D = librosa.stft(
            self.audio_data,
            n_fft=self.window_size_spin.value(),
            hop_length=self.hop_size_spin.value(),
            win_length=self.window_size_spin.value()
        )
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(
            S_db,
            sr=self.sample_rate,
            hop_length=self.hop_size_spin.value(),
            x_axis='time',
            y_axis='linear',
            ax=ax2,
            cmap='viridis',
            vmin=-60,
            vmax=0
        )
        ax2.set_ylim(0, self.max_freq_spin.value())
        self.playback_lines.append(ax2.axvline(x=0, color='r', linewidth=1, animated=True))
        ax2.set_title("Spectrogram")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlabel("Time (s)")
        
        # Draw everything once
        self.canvas.draw()
        
        # Store the background without the cursor lines
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        
        # Mark the cursor lines as animated
        for line in self.playback_lines:
            line.set_animated(True)

    def update_playback_cursor(self, position):
        current_time = position / 1000  # Convert ms to seconds
        
        # Check if we've reached the end (with a small buffer)
        if position >= self.media_player.duration() - 100:  # 100ms buffer
            self.play_btn.setText("Play")
        
        # Update cursor positions
        for line in self.playback_lines:
            line.set_xdata([current_time, current_time])
        
        # Only proceed if we have a background
        if not hasattr(self, 'background'):
            self.canvas.draw()
            return
        
        try:
            # Restore the background
            self.canvas.restore_region(self.background)
            
            # Redraw each cursor line
            for line in self.playback_lines:
                line.axes.draw_artist(line)
            
            # Blit the updated regions
            self.canvas.blit(self.figure.bbox)
            self.canvas.flush_events()
        except Exception as e:
            print(f"Error during blitting: {e}")
            # Fallback to full redraw
            self.canvas.draw()

    def toggle_playback(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_btn.setText("Play")
        else:
            if self.media_player.position() >= self.media_player.duration() - 100:
                self.media_player.setPosition(0)

            # Check if it's the first playback
            if self.first_playback:
                self.plot_spectrogram()
                self.first_playback = False
                
            self.media_player.play()
            self.play_btn.setText("Pause")

    def stop_playback(self):
        self.media_player.stop()
        self.play_btn.setText("Play")
        self.update_playback_cursor(0)

   


