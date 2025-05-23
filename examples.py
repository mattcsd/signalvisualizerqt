import numpy as np
import librosa
import librosa.display
import os
import time
import matplotlib.pyplot as plt
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtWidgets import ( QTextBrowser, QComboBox, QCheckBox, QWidget, QVBoxLayout, QLabel, QScrollArea, 
                            QGroupBox, QPushButton, QMessageBox, 
                            QFormLayout, QSpinBox, QHBoxLayout,
                            QSizePolicy, QApplication)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from scipy.signal import find_peaks
from scipy.signal.windows import blackmanharris 

# Add this import at the top
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl

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

        # Add FFT parameters initialization
        self.fft_size = 4096 * 4  # 32768-point FFT for high resolution
        self.peak_markers = None  # Will be initialized in plot_spectrog
        
        # Get recordings directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.recordings_dir = os.path.join(current_dir, "recordings")
        os.makedirs(self.recordings_dir, exist_ok=True)  # Create directory if it doesn't exist
        
        # Media player setup
        self.media_player = QMediaPlayer()
        self.media_player.setNotifyInterval(20)
        self.media_player.positionChanged.connect(self.update_playback_cursor)
        
        self.init_ui()
        self.load_audio_files_list()  # Load available audio files

    def init_ui(self):
        self.setWindowTitle("Beat Frequency Visualizer")
        self.setMinimumSize(1000, 800)
        
        # Initialize widgets first
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
        
        # File selection dropdown
        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(200)
        self.file_combo.currentIndexChanged.connect(self.on_file_selected)
        
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)
        
        # Parameters layout
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Max Freq (Hz):"))
        param_layout.addWidget(self.max_freq_spin)
        param_layout.addWidget(QLabel("Window Size:"))
        param_layout.addWidget(self.window_size_spin)
        param_layout.addWidget(QLabel("Hop Size:"))
        param_layout.addWidget(self.hop_size_spin)
        param_layout.addWidget(self.replot_btn)

        # Add Help button
        self.help_btn = QPushButton("Help")
        self.help_btn.clicked.connect(self.show_help)
        param_layout.addWidget(self.help_btn)

        
        # Playback layout
        playback_layout = QHBoxLayout()
        playback_layout.addWidget(QLabel("Select File:"))
        playback_layout.addWidget(self.file_combo)
        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.stop_btn)
        
        # Add to control panel
        control_layout.addLayout(param_layout)
        control_layout.addStretch()
        control_layout.addLayout(playback_layout)
        
        # Visualization area
        self.figure = Figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # Add widgets to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas, stretch=1)
        
        self.setLayout(main_layout)

    def show_help(self):
        """Show help documentation for the current audio phenomenon"""
        current_file = self.file_combo.currentText().lower()
        
        # Create HTML content based on the selected file
        html_content = self.generate_help_content(current_file)
        
        # Create help window
        self.help_window = QWidget()
        self.help_window.setWindowTitle(f"Help: {current_file}")
        self.help_window.resize(800, 600)
        
        layout = QVBoxLayout()
        web_view = QWebEngineView()
        
        # Use a data URL to display our HTML content
        web_view.setHtml(html_content, QUrl.fromLocalFile(''))
        
        layout.addWidget(web_view)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.help_window.close)
        layout.addWidget(close_btn)
        
        self.help_window.setLayout(layout)
        self.help_window.show()

    def plot_spectrogram(self):
        if self.audio_data is None:
            return
            
        self.figure.clear()
        self.playback_lines = []
        
        # Create 4 subplots and store their references
        gs = self.figure.add_gridspec(4, 1, height_ratios=[1, 1, 2, 2], hspace=0.6)
        
        # 1. Waveform plot
        self.ax_wave = self.figure.add_subplot(gs[0])
        self.ax_wave.plot(self.time, self.audio_data, color='b', linewidth=0.5, alpha=0.7)
        self.playback_lines.append(self.ax_wave.axvline(x=0, color='r', linewidth=1, animated=True))
        
        # 2. Amplitude envelope plot
        self.ax_env = self.figure.add_subplot(gs[1], sharex=self.ax_wave)
        amplitude = np.abs(self.audio_data)
        smooth_window = int(0.02 * self.sample_rate)
        amplitude_smooth = np.convolve(amplitude, np.ones(smooth_window)/smooth_window, mode='same')
        self.ax_env.plot(self.time, amplitude_smooth, 'b-', linewidth=1)
        self.playback_lines.append(self.ax_env.axvline(x=0, color='r', linewidth=1, animated=True))
        
        # 3. Spectrogram plot
        self.ax_spec = self.figure.add_subplot(gs[2], sharex=self.ax_wave)
        D = librosa.stft(self.audio_data,
                        n_fft=self.window_size_spin.value(),
                        hop_length=self.hop_size_spin.value(),
                        win_length=self.window_size_spin.value())
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db,
                               sr=self.sample_rate,
                               hop_length=self.hop_size_spin.value(),
                               x_axis='time',
                               y_axis='linear',
                               ax=self.ax_spec,
                               cmap='viridis',
                               vmin=-60,
                               vmax=0)
        self.playback_lines.append(self.ax_spec.axvline(x=0, color='r', linewidth=1, animated=True))
        
        # 4. Real-time FFT plot
        self.ax_fft = self.figure.add_subplot(gs[3])
        freqs = np.fft.rfftfreq(self.fft_size, 1/self.sample_rate)  # Use self.fft_size

        self.fft_line, = self.ax_fft.semilogx(freqs, np.zeros_like(freqs), 'b-', linewidth=0.8)
        self.peak_markers, = self.ax_fft.plot([], [], 'ro', markersize=4, alpha=0.7)

        # Enhanced FFT plot styling
        self.ax_fft.set_title("High-Resolution Frequency Spectrum", pad=8)
        self.ax_fft.set_xlim(20, 10000)
        self.ax_fft.set_ylim(-80, 0)  # Wider dynamic range
        self.ax_fft.grid(True, which='both', alpha=0.3)
        self.ax_fft.set_xlabel("Frequency (Hz)", fontsize=9)
        self.ax_fft.set_ylabel("Magnitude (dB)", fontsize=9)


        # Set titles and labels for all axes
        self.ax_wave.set_title("Waveform")
        self.ax_env.set_title("Amplitude Envelope")
        self.ax_spec.set_title("Spectrogram")
        self.ax_fft.set_title("Real-time FFT")
        
        # Set axis limits and labels
        self.ax_wave.set_xlim(0, self.time[-1])
        self.ax_spec.set_ylim(0, self.max_freq_spin.value())
        self.ax_fft.set_xlim(20, 10000)
        self.ax_fft.set_ylim(-60, 0)
        self.figure.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Draw everything
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        
        for line in self.playback_lines:
            line.set_animated(True)

    def update_playback_cursor(self, position):
        current_time = position / 1000  # Convert ms to seconds
        
        # Update real-time FFT only when playing
        if self.media_player.state() == QMediaPlayer.PlayingState:
            start_sample = int(current_time * self.sample_rate)
            end_sample = start_sample + self.window_size_spin.value()
            
            if end_sample < len(self.audio_data):
                frame = self.audio_data[start_sample:end_sample]
                
                # Apply window
                window = blackmanharris(len(frame))
                frame_windowed = frame * window
                
                # Compute FFT
                fft_result = np.fft.rfft(frame_windowed, n=self.fft_size)
                fft_magnitude = np.abs(fft_result)
                fft_magnitude_db = 20 * np.log10(fft_magnitude + 1e-8)
                fft_magnitude_db -= np.max(fft_magnitude_db)  # Normalize so 0 dB is peak
                fft_magnitude_db = np.clip(fft_magnitude_db, -60, 0)


                # Update plot
                freqs = np.fft.rfftfreq(self.fft_size, 1/self.sample_rate)
                self.fft_line.set_data(freqs, fft_magnitude_db) 

                # Find and mark peaks
                peaks, _ = find_peaks(fft_magnitude_db, height=-40, prominence=6, width=2)
                if len(peaks) > 0:
                    self.peak_markers.set_data(freqs[peaks], fft_magnitude_db[peaks])
                else:
                    self.peak_markers.set_data([], [])

                        
        # Update playback cursors for other plots
        for line in self.playback_lines:
            line.set_xdata([current_time, current_time])
        
        # Handle the blitting for smooth updates
        if hasattr(self, 'background'):
            try:
                # Restore background
                self.canvas.restore_region(self.background)
                
                # Redraw FFT plot if playing
                if self.media_player.state() == QMediaPlayer.PlayingState:
                    self.ax_fft.draw_artist(self.fft_line)
                
                # Redraw cursor lines
                for line in self.playback_lines:
                    line.axes.draw_artist(line)
                
                # Blit the updated regions
                self.canvas.blit(self.figure.bbox)
            except Exception as e:
                print(f"Blitting error: {e}")
                # Fallback to full redraw if blitting fails
                self.canvas.draw()

    def load_audio_files_list(self):
        """Load all WAV files from the recordings directory into the dropdown"""
        try:
            self.file_combo.clear()
            files = [f for f in os.listdir(self.recordings_dir) if f.lower().endswith('.wav')]
            
            if not files:
                self.file_combo.addItem("No WAV files found")
                return
            
            for file in sorted(files):
                self.file_combo.addItem(file)
            
            # Try to select beat.wav by default if it exists
            beat_index = self.file_combo.findText("beat.wav")
            if beat_index >= 0:
                self.file_combo.setCurrentIndex(beat_index)
            else:
                self.on_file_selected(0)  # Load first file by default
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not list audio files: {str(e)}")

    def on_file_selected(self, index):
        """Handle when a new file is selected from the dropdown"""
        if self.file_combo.count() == 0 or self.file_combo.currentText() == "No WAV files found":
            return
            
        selected_file = self.file_combo.currentText()
        file_path = os.path.join(self.recordings_dir, selected_file)
        
        try:
            # Stop any current playback and reset button
            self.media_player.stop()
            self.play_btn.setText("Play")  # <-- Add this line
            
            self.audio_data, self.sample_rate = librosa.load(
                file_path, 
                sr=44100,
                mono=True
            )
            self.time = np.arange(len(self.audio_data)) / self.sample_rate
            
            url = QUrl.fromLocalFile(file_path)
            self.media_player.setMedia(QMediaContent(url))
            
            self.plot_spectrogram()
            self.first_playback = True  # Reset flag for new file
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio: {str(e)}")

        
    def cleanup(self):
        self.media_player.stop()
        self.media_player.setMedia(QMediaContent())  # Clear media
        # Clear any other resources if needed

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

    def generate_help_content(self, filename):
        """Generate HTML help content based on the audio file"""
        filename_lower = filename.lower()
        
        if "beat" in filename_lower:
            return """
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    h1 { color: #2c3e50; }
                    img { max-width: 100%; height: auto; display: block; margin: 20px auto; }
                    .note { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; }
                </style>
            </head>
            <body>
                <h1>Beat Frequencies in Acoustics</h1>
                
                <p>Beat frequencies occur when two sound waves of slightly different frequencies interfere with each other.</p>
                
                <h2>What Musicians Should Know</h2>
                
                <p>When tuning instruments, beats are actually useful! Here's why:</p>
                
                <div class="note">
                    <strong>Practical Tip:</strong> When tuning two strings to unison, listen for beats. 
                    As the frequencies get closer, the beats slow down. When they completely disappear, 
                    your strings are in perfect tune!
                </div>
                
                <h3>The Science Behind Beats</h3>
                
                <p>Mathematically, when two waves with frequencies f₁ and f₂ combine, you hear:</p>
                
                <ul>
                    <li>A <strong>carrier frequency</strong> at the average: (f₁ + f₂)/2</li>
                    <li>A <strong>beat frequency</strong> at the difference: |f₁ - f₂|</li>
                </ul>
                
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Interference_of_two_waves.svg/1200px-Interference_of_two_waves.svg.png" 
                     alt="Wave interference diagram" style="max-width: 500px;">
                <p style="text-align: center;"><em>Constructive and destructive interference creates the beating effect</em></p>
                
                <h3>Musical Applications</h3>
                
                <ul>
                    <li>Tuning instruments (especially strings and pipe organs)</li>
                    <li>Creating special effects in electronic music</li>
                    <li>Understanding how vibrato works</li>
                </ul>
                
                <h3>Try This Experiment</h3>
                
                <p>In this visualization:</p>
                <ol>
                    <li>Look at the waveform - see the amplitude modulation?</li>
                    <li>Check the spectrogram - can you spot the two close frequencies?</li>
                    <li>Listen carefully - count how many beats occur per second</li>
                </ol>
            </body>
            </html>
            """
        
        elif "combo" in filename_lower:
            return """
            <html><body style="font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #2c3e50;">Resonating Frequencies</h1>
                <p>This demonstrates how open strings resonate when other strings are played.</p>
                
                <h2>What's Happening</h2>
                <ul>
                    <li>First pluck shows the fundamental frequency</li>
                    <li>Subsequent plucks show sympathetic vibrations in other strings</li>
                    <li>Particularly strong on strings that share harmonic relationships</li>
                </ul>
                
                <h2>Musical Uses</h2>
                <ul>
                    <li><strong>Resonance:</strong> Harp and piano use this effect deliberately</li>
                    <li><strong>Tuning:</strong> Helps verify proper string relationships</li>
                    <li><strong>Composition:</strong> Used in spectral music compositions</li>
                </ul>
            </body></html>
            """
        
        elif "harmonics" in filename_lower:
            return """
            <html><body style="font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #2c3e50;">String Harmonics</h1>
                <p>Harmonics are pure tones created by dividing the string at specific nodes.</p>
                
                <h2>Types Demonstrated</h2>
                <ul>
                    <li><strong>Natural harmonics:</strong> Created by lightly touching the string at fractional divisions</li>
                    <li><strong>Artificial harmonics:</strong> Created by stopping the string and touching a node</li>
                </ul>
                
                <div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #3498db;">
                    <strong>Did You Know?</strong> The harmonic at the 12th fret is exactly one octave above the open string.
                </div>
                
                <h2>Practical Uses</h2>
                <ul>
                    <li>Tuning reference (harmonics are always perfectly in tune)</li>
                    <li>Special effects in compositions</li>
                    <li>Checking instrument intonation</li>
                </ul>
            </body></html>
            """
        
        elif "eigen" in filename_lower:
            return """
            <html><body style="font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #2c3e50;">Instrument Eigenvalues</h1>
                <p>These are the natural resonant frequencies of your instrument's body.</p>
                
                <h2>What You're Hearing</h2>
                <ul>
                    <li>The "voice" of your instrument's wood and construction</li>
                    <li>Peak resonance frequencies that amplify certain notes</li>
                    <li>Characteristics that make each instrument unique</li>
                </ul>
                
                <h2>Why It Matters</h2>
                <ul>
                    <li><strong>Luthiers:</strong> Use this to fine-tune instrument construction</li>
                    <li><strong>Players:</strong> Helps identify an instrument's "sweet spots"</li>
                    <li><strong>Recording:</strong> Knowing these helps with microphone placement</li>
                </ul>
            </body></html>
            """
        
        elif "tapedelay" in filename_lower:
            return """
            <html><body style="font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #2c3e50;">Analog Tape Delay Effects</h1>
                <p>This demonstrates the warm, organic sound of analog delay circuits.</p>
                
                <h2>Characteristics</h2>
                <ul>
                    <li>High-frequency rolloff creates warmer repeats</li>
                    <li>Slight modulation adds "wobble" to the sound</li>
                    <li>Non-linear saturation creates musical distortion</li>
                </ul>
                
                <div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #3498db;">
                    <strong>Pro Tip:</strong> Set delay time to match song tempo (60000/BPM = ms delay time)
                </div>
                
                <h2>Musical Uses</h2>
                <ul>
                    <li><strong>Slapback:</strong> Short delays for rockabilly and blues</li>
                    <li><strong>Ambience:</strong> Longer delays for atmospheric effects</li>
                    <li><strong>Rhythm:</strong> Syncopated delays for reggae and dub</li>
                </ul>
            </body></html>
            """
        
        else:
            return """
            <html><body style="font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #2c3e50;">Audio Analysis Help</h1>
                <p>Use this visualizer to explore:</p>
                <ul>
                    <li><strong>Waveform:</strong> See the amplitude changes over time</li>
                    <li><strong>Spectrogram:</strong> View frequency content evolution</li>
                    <li><strong>FFT:</strong> Analyze harmonic content at any moment</li>
                </ul>
                <p>Try adjusting the analysis parameters to see different aspects of the sound.</p>
            </body></html>
            """


