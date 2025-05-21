from PyQt5.QtWidgets import ( QSlider, QHBoxLayout, QDialog, QLabel, QPushButton, QLineEdit, QRadioButton, 
                            QCheckBox, QComboBox, QGridLayout, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector, Cursor, MultiCursor
import numpy as np
import sounddevice as sd
import librosa
from scipy import signal
import matplotlib as mpl
from pitchAdvancedSettings import AdvancedSettings
from PyQt5.QtWidgets import QVBoxLayout
from scipy.io.wavfile import write
from help import Help
from pathlib import Path
import time
import matplotlib.gridspec as gridspec


class ControlMenu(QDialog):
    def __init__(self, name, fs, audio, duration, controller):
        super().__init__(None)
        self.base_name = name.split('_[')[0] if '_[' in name else name.split(' [')[0] if ' [' in name else name        
        self.audio = audio
        self.current_audio = audio  # Will hold either full audio or selection
        self.fs = fs
        self.duration = duration
        self.lenAudio = len(audio)
        self.time = np.arange(0, self.lenAudio/self.fs, 1/self.fs)
        self.controller = controller
        self.current_figure = None
        self.selected_span = None  # Initialize as None

        self.plot_windows = []  # Track plot windows
        self.span_selectors = {}  # Track span selectors by window ID
        
        # CRITICAL SETTINGS (add these):
        self.setWindowTitle(self.base_name)  # Use the provided title

        self.setMinimumSize(800, 600)  # Force reasonable size
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.plot_windows = []  # Track all plot windows
        self.selected_span = None  # Track span selection times

        self.audio_player = None
        self.is_playing = False


        np.seterr(divide='ignore')
        self.span = None
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle(self.base_name)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        main_layout = QGridLayout()
        main_layout.setVerticalSpacing(8)
        main_layout.setHorizontalSpacing(10)
        
        self.method_selector = QComboBox()
        self.method_selector.addItems([
            'FT', 'STFT', 'Spectrogram', 'STFT + Spect', 
            'Short-Time-Energy', 'Pitch', 'Spectral Centroid', 'Filtering'
        ])
        self.method_selector.setCurrentText('Spectrogram')
        self.method_selector.currentTextChanged.connect(self.update_ui_state)
        main_layout.addWidget(QLabel('Analysis Method:'), 0, 0)
        main_layout.addWidget(self.method_selector, 0, 1, 1, 3)
        
        self.create_spectrogram_group(main_layout)
        self.create_pitch_group(main_layout)
        self.create_filter_group(main_layout)
        
        self.filter_response_button = QPushButton('Show Filter Response')
        self.filter_response_button.clicked.connect(self.plot_filter_response)
        main_layout.addWidget(self.filter_response_button, 8, 2, 1, 2)
        
        self.create_ste_group(main_layout)
        
        self.plot_button = QPushButton('Plot')
        self.plot_button.clicked.connect(self.plot_figure)
        main_layout.addWidget(self.plot_button, 14, 3, 1, 1)
     
        self.help_button = QPushButton('🛈')
        self.help_button.setFixedWidth(30)
        self.help_button.clicked.connect(self.show_help)
        main_layout.addWidget(self.help_button, 14, 2, 1, 1, Qt.AlignRight)
        
        self.setLayout(main_layout)
        self.update_ui_state('Spectrogram')

    def play_from_current_position(self):
        """Stream audio from the current position using a low-latency callback."""

        self.stop_audio()

        self.mid_point_idx = 0
        self.playback_start_sample = 0

        # Use full audio
        audio_chunk = self.audio

        if len(audio_chunk) == 0:
            print("Empty audio_chunk — skipping playback")
            return

        if np.max(np.abs(audio_chunk)) > 1.0:
            audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))

        if audio_chunk.ndim > 1:
            print(f"Converting multichannel audio with shape {audio_chunk.shape} to mono.")
            audio_chunk = np.mean(audio_chunk, axis=1)

        if not np.any(audio_chunk):
            print("Audio chunk is silent or empty!")
            return

        print(f"Audio length: {len(audio_chunk)}, sample rate: {self.fs}")

        self.stream_pos = 0
        self.stream_data = audio_chunk.astype(np.float32)
        total_length = len(self.stream_data)

        def callback(outdata, frames, time, status):
            if status:
                print("Stream status:", status)

            remaining = total_length - self.stream_pos
            if remaining <= 0:
                raise sd.CallbackStop()

            n_frames = min(frames, remaining)
            chunk = self.stream_data[self.stream_pos:self.stream_pos + n_frames]

            if n_frames < frames:
                chunk = np.pad(chunk, (0, frames - n_frames))

            outdata[:, 0] = chunk
            self.stream_pos += n_frames
            self.mid_point_idx = self.stream_pos  # <-- live cursor index for update_playback_position()

            if self.stream_pos >= total_length:
                raise sd.CallbackStop()

        try:
            self.active_stream = sd.OutputStream(
                samplerate=self.fs,
                channels=1,
                dtype='float32',
                callback=callback,
                blocksize=4096,
            )
            self.active_stream.start()
            self.is_playing = True
            self.playback_start_time = time.time()
            self.playback_start_sample = self.stream_pos

            # Start visualization timer
            if not hasattr(self, 'playback_timer'):
                self.playback_timer = QTimer()
                self.playback_timer.timeout.connect(self.update_playback_position)
            self.playback_timer.start(5)

        except Exception as e:
            print(f"Audio playback failed: {e}")


    def show_help(self):
        """Properly shows and activates the help window"""
        if hasattr(self.controller, 'help'):
            # Ensure the help window is properly parented
            if not self.controller.help.parent():
                self.controller.help.setParent(self)
                
            # Force the window to appear in front
            self.controller.help.setWindowFlags(
                self.controller.help.windowFlags() | 
                Qt.WindowStaysOnTopHint
            )
            self.controller.help.show()
            self.controller.help.raise_()
            self.controller.help.activateWindow()
            
            # Remove the always-on-top hint after showing
            self.controller.help.setWindowFlags(
                self.controller.help.windowFlags() & 
                ~Qt.WindowStaysOnTopHint
            )
            self.controller.help.show()
            
            # Load the help content
            self.controller.help.createHelpMenu(8)
        else:
            QMessageBox.warning(self, "Help", "Help system not available")

    def format_timestamp(self, seconds):
        """Convert seconds to mm:ss.ms format with exactly 2 decimal places"""
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"[:8]  # Format as 00:00.00

    def update_plot_window_titles(self):
        """Update all open plot windows with current selection"""
        if not hasattr(self, 'selected_span') or not self.selected_span:
            return
        
        method = self.method_selector.currentText()
        start, end = self.selected_span
        new_title = f"{method}_{self.base_name}_{self.format_timestamp(start)}-{self.format_timestamp(end)}"
        
        for window in self.plot_windows:
            try:
                if window.isVisible():
                    window.setWindowTitle(new_title)
            except RuntimeError:
                continue  # Skip deleted windows

    def create_spectrogram_group(self, layout):
        group = QGroupBox("Spectrogram")
        grid = QGridLayout()
        
        self.window_type = QComboBox()
        self.window_type.addItems(['Bartlett', 'Blackman', 'Hamming', 'Hanning', 'Kaiser'])
        
        self.window_size = QLineEdit()
        self.window_size.setValidator(QDoubleValidator(0.01, self.duration, 3))
        if self.duration <= 0.03:
            self.window_size.setText(f"{self.duration - 0.001:.3f}")
        else:
            self.window_size.setText("0.03")
            
        self.overlap = QLineEdit()
        self.overlap.setValidator(QDoubleValidator(0.0, self.duration, 3))
        if self.duration <= 0.03:
            wind_size = float(self.window_size.text())
            self.overlap.setText(f"{wind_size - 0.001:.3f}")
        else:
            self.overlap.setText("0.01")
            
        self.nfft = QComboBox()
        nfft_values = [2**i for i in range(9, 20)]
        self.nfft.addItems(map(str, nfft_values))
        self.nfft.setCurrentText("2048")
        
        self.min_freq = QLineEdit("0")
        self.max_freq = QLineEdit(str(self.fs//2))
        
        self.draw_style = QComboBox()
        self.draw_style.addItems(["Linear", "Mel"])
        
        self.show_pitch = QCheckBox("Show Pitch")
        self.show_pitch.stateChanged.connect(self.toggle_pitch_controls)
        
        grid.addWidget(QLabel("Window:"), 0, 0)
        grid.addWidget(self.window_type, 0, 1)
        grid.addWidget(QLabel("Window size (s):"), 1, 0)
        grid.addWidget(self.window_size, 1, 1)
        grid.addWidget(QLabel("Overlap (s):"), 2, 0)
        grid.addWidget(self.overlap, 2, 1)
        grid.addWidget(QLabel("NFFT:"), 3, 0)
        grid.addWidget(self.nfft, 3, 1)
        grid.addWidget(QLabel("Min freq (Hz):"), 4, 0)
        grid.addWidget(self.min_freq, 4, 1)
        grid.addWidget(QLabel("Max freq (Hz):"), 5, 0)
        grid.addWidget(self.max_freq, 5, 1)
        grid.addWidget(QLabel("Drawing style:"), 6, 0)
        grid.addWidget(self.draw_style, 6, 1)
        grid.addWidget(self.show_pitch, 7, 0, 1, 2)
        
        group.setLayout(grid)
        layout.addWidget(group, 1, 0, 8, 2)

    def toggle_live_analysis(self):
        """Toggle live analysis mode for STFT windows"""
        # Get which button was clicked
        sender = self.sender()
        
        if sender.isChecked():
            # Button was toggled ON
            sender.setText("⏹ Stop Live Analysis")
            self.start_live_analysis()
        else:
            # Button was toggled OFF
            sender.setText("▶ Start Live Analysis")
            self.stop_live_analysis()


    def start_live_analysis(self):
        """Start live analysis for STFT windows"""
        # Get the button that triggered this
        sender = self.sender()
        self.live_analysis_btn = sender

        # Find the parent plot dialog
        parent = sender.parent()
        while parent and not isinstance(parent, QDialog):
            parent = parent.parent()
        
        if not parent:
            print("Couldn't find STFT plot window")
            return
        
        # Store reference to current window
        self.current_live_window = parent
        
        # Initialize analysis
        if not hasattr(self, 'live_timer'):
            self.live_timer = QTimer()
            self.live_timer.timeout.connect(self.update_live_position)
        
        # Start audio playback
        self.play_from_current_position()
        
        # Start timer
        self.live_timer.start(100)

    def update_playback_position(self):
        # Use QMediaPlayer's position directly instead of time.time()
        if not hasattr(self, 'media_player') or not self.media_player:
            return

        elapsed_ms = self.media_player.position()  # milliseconds
        self.mid_point_idx = min(len(self.audio) - 1, int((elapsed_ms / 1000.0) * self.fs))

        # Update waveform and STFT windows
        if hasattr(self, 'current_figure'):
            try:
                ax1, ax2, ax3 = self.current_figure.axes[:3]
            except ValueError:
                print("Expected 3 axes in current_figure, got something else")
                return
            
            # Current window around the cursor
            #check if not fixed
            window_size = 1024
            start_idx = max(0, self.mid_point_idx - window_size // 2)
            end_idx = min(len(self.audio), start_idx + window_size)
            segment = self.audio[start_idx:end_idx]

            if len(segment) > 0:
                self.update_stft_spect_plot(ax1, ax2, ax3, segment=segment)

        # Check if playback has reached end
        if self.mid_point_idx >= len(self.audio):
            self.stop_audio_playback()
            if hasattr(self, 'live_analysis_button'):
                self.live_analysis_button.setText("▶ Play Audio")

        print(f"Updating position: mid_point_idx = {self.mid_point_idx}")



    def start_audio_playback(self):
        """Start audio playback from current cursor position"""

        self.mid_point_idx = 0
        self.playback_start_sample = 0
        # Calculate start position
        start_sample = int(self.mid_point_idx)
        audio_segment = self.audio[start_sample:]
        
        # Normalize if needed
        if np.max(np.abs(audio_segment)) > 1.0:
            audio_segment = audio_segment / np.max(np.abs(audio_segment))
        
        # Start playback
        sd.stop()  # Stop any existing playback
        self.audio_player = sd.play(audio_segment, self.fs, blocking=False)
        self.is_playing_audio = True
        
        # Store playback info
        self.playback_start_time = time.time()
        self.playback_start_sample = start_sample
        
        # Start position updater if needed
        if not hasattr(self, 'playback_timer'):
            self.playback_timer = QTimer()
            self.playback_timer.timeout.connect(self.update_playback_position)
        self.playback_timer.start(16)  # Update every 50ms

    def stop_audio_playback(self):
        """Stop any ongoing audio playback"""
        import sounddevice as sd
        sd.stop()
        self.is_playing_audio = False
        if hasattr(self, 'playback_timer'):
            self.playback_timer.stop()

    def play_audio_segment(self, start_sample, end_sample):
        """Play a segment of audio without blocking"""
        import sounddevice as sd
        segment = self.audio[start_sample:end_sample]
        
        # Basic audio normalization
        if np.max(np.abs(segment)) > 1.0:
            segment = segment / np.max(np.abs(segment))
        
        # Store playback info
        self.current_playback = sd.play(segment, self.fs, blocking=False)
        self.playback_start_time = time.time()
        self.playback_start_sample = start_sample

    def stop_all_audio(self):
        try:
            if hasattr(self, 'active_stream') and self.active_stream:
                self.active_stream.stop()
                self.active_stream.close()
                self.active_stream = None
        except Exception as e:
            print(f"Error stopping audio: {e}")

    def stop_live_analysis(self):
        """Stop live analysis for STFT windows"""
        # Stop audio first
        self.stop_audio()
        
        # Stop timer if it exists
        if hasattr(self, 'live_timer'):
            try:
                self.live_timer.stop()
            except:
                pass
        
        # Update all STFT window buttons
        for window in self.plot_windows:
            if hasattr(window, 'live_analysis_btn'):
                window.live_analysis_btn.setChecked(False)
                window.live_analysis_btn.setText("▶ Start Live Analysis")

    def stop_audio(self):
        """Stop all audio playback"""
        import sounddevice as sd
        if hasattr(self, 'is_playing') and self.is_playing:
            sd.stop()
            self.is_playing = False
        if hasattr(self, 'playback_timer'):
            self.playback_timer.stop()

    def update_live_position(self):
        """Safe position update with window checks"""
        if not hasattr(self, 'wind_size_samples'):
            return
        
        # Check if window still exists
        if not hasattr(self, 'current_figure') or not plt.fignum_exists(self.current_figure.number):
            self.stop_live_analysis()
            return
        
        # Update position
        elapsed = time.time() - self.playback_start_time
        self.mid_point_idx = min(len(self.audio) - 1, self.playback_start_sample + int(elapsed * self.fs))        
        

        try:
            # Safe visualization update
            ax1, ax2, ax3 = self.current_figure.axes[:3]
            self.update_stft_spect_plot(ax1, ax2, ax3)
        except (AttributeError, RuntimeError) as e:
            print(f"Visual update failed: {e}")
            self.stop_live_analysis()
        
        # Auto-stop at end
        if self.mid_point_idx >= len(self.audio) - self.wind_size_samples//2:
            self.stop_live_analysis()

    def create_pitch_group(self, layout):
        group = QGroupBox("Pitch")
        grid = QGridLayout()
        
        self.pitch_method = QComboBox()
        self.pitch_method.addItems(['Autocorrelation', 'Cross-correlation', 'Subharmonics', 'Spinet'])
        
        self.min_pitch = QLineEdit("75.0")
        self.max_pitch = QLineEdit("600.0")
        
        self.adv_settings = QPushButton("Advanced Settings")
        self.adv_settings.clicked.connect(self.controller.adse.advancedSettings)
       
        grid.addWidget(QLabel("Method:"), 0, 0)
        grid.addWidget(self.pitch_method, 0, 1)
        grid.addWidget(QLabel("Min pitch (Hz):"), 1, 0)
        grid.addWidget(self.min_pitch, 1, 1)
        grid.addWidget(QLabel("Max pitch (Hz):"), 2, 0)
        grid.addWidget(self.max_pitch, 2, 1)
        grid.addWidget(self.adv_settings, 3, 0, 1, 2)
        
        group.setLayout(grid)
        layout.addWidget(group, 9, 0, 5, 2)

    def toggle_pitch_controls(self, state):
        enabled = state == Qt.Checked
        self.pitch_method.setEnabled(enabled)
        self.min_pitch.setEnabled(enabled)
        self.max_pitch.setEnabled(enabled)
        self.adv_settings.setEnabled(enabled)

    def show_advanced_settings(self):
        dialog = AdvancedSettings(self)
        if dialog.exec_() == QDialog.Accepted:
            autocorr_vars = dialog.getAutocorrelationVars()
            subharmonic_vars = dialog.getSubharmonicsVars()
            spinet_vars = dialog.getSpinetVars()
            other_vars = dialog.getVariables()

            self.controller.adse = {
                'autocorr': autocorr_vars,
                'subharmonics': subharmonic_vars,
                'spinet': spinet_vars,
                'other': other_vars
            }
            
    def create_filter_group(self, layout):
        group = QGroupBox("Filtering")
        grid = QGridLayout()
        
        self.filter_type = QComboBox()
        self.filter_type.addItems(['Harmonic', 'Lowpass', 'Highpass', 'Bandpass', 'Bandstop'])
        self.filter_type.currentTextChanged.connect(self.update_filter_ui)
        
        self.fund_freq = QLineEdit("1")
        self.center_freq = QLineEdit("400")
        self.percentage = QLineEdit("10.0")
        self.fcut = QLineEdit("1000")
        self.fcut1 = QLineEdit("200")
        self.fcut2 = QLineEdit("600")
        
        grid.addWidget(QLabel("Type:"), 0, 0)
        grid.addWidget(self.filter_type, 0, 1)
        grid.addWidget(QLabel("Fund. freq. mult:"), 1, 0)
        grid.addWidget(self.fund_freq, 1, 1)
        grid.addWidget(QLabel("Center freq:"), 2, 0)
        grid.addWidget(self.center_freq, 2, 1)
        grid.addWidget(QLabel("Percentage (%):"), 3, 0)
        grid.addWidget(self.percentage, 3, 1)
        grid.addWidget(QLabel("Fcut:"), 4, 0)
        grid.addWidget(self.fcut, 4, 1)
        grid.addWidget(QLabel("Fcut1:"), 5, 0)
        grid.addWidget(self.fcut1, 5, 1)
        grid.addWidget(QLabel("Fcut2:"), 6, 0)
        grid.addWidget(self.fcut2, 6, 1)


        # Add visualization type radio buttons
        self.vis_type_label = QLabel("Visualization:")
        self.waveform_radio = QRadioButton("Waveform")
        self.spectrogram_radio = QRadioButton("Spectrogram")
        self.waveform_radio.setChecked(True)  # Default to waveform

        # In your initialization code where you create the radio buttons:
        self.waveform_radio.toggled.connect(lambda: self.update_ui_state('Filtering'))
        self.spectrogram_radio.toggled.connect(lambda: self.update_ui_state('Filtering'))

        
        # Add to layout (adjust row numbers as needed)
        grid.addWidget(self.vis_type_label, 7, 0)
        grid.addWidget(self.waveform_radio, 7, 1)
        grid.addWidget(self.spectrogram_radio, 8, 1)
        
        group.setLayout(grid)
        layout.addWidget(group, 1, 2, 7, 2)
        
    def update_filter_ui(self, filter_type):
        harmonic = filter_type == 'Harmonic'
        lphp = filter_type in ['Lowpass', 'Highpass']
        bpbs = filter_type in ['Bandpass', 'Bandstop']
        
        self.fund_freq.setEnabled(harmonic)
        self.center_freq.setEnabled(harmonic)
        self.percentage.setEnabled(True)
        self.fcut.setEnabled(lphp)
        self.fcut1.setEnabled(bpbs)
        self.fcut2.setEnabled(bpbs)
        
    def create_ste_group(self, layout):
        group = QGroupBox("Short-Time Energy")
        grid = QGridLayout()
        
        self.beta = QLineEdit("0")
        
        grid.addWidget(QLabel("Beta:"), 0, 0)
        grid.addWidget(self.beta, 0, 1)
        
        group.setLayout(grid)
        layout.addWidget(group, 9, 2, 5, 2)
        
    def update_ui_state(self, method):
        # First define all the mode flags
        ft_enabled = method == 'FT'
        stft_enabled = method == 'STFT'
        spectrogram_enabled = method == 'Spectrogram'
        stft_spect_enabled = method == 'STFT + Spect'
        ste_enabled = method == 'Short-Time-Energy'
        pitch_enabled = method == 'Pitch'
        spectral_centroid_enabled = method == 'Spectral Centroid'
        filtering_enabled = method == 'Filtering'
        
        # Determine if we're showing spectrograms in filtering mode
        show_filtering_spectrograms = (filtering_enabled and 
                                      hasattr(self, 'waveform_radio') and 
                                      not self.waveform_radio.isChecked())
        
        # Spectrogram controls
        self.window_type.setEnabled(stft_enabled or spectrogram_enabled or stft_spect_enabled or 
                                  spectral_centroid_enabled or ste_enabled or show_filtering_spectrograms)
        self.window_size.setEnabled(not ft_enabled and not pitch_enabled and 
                                  (not filtering_enabled or show_filtering_spectrograms))
        self.overlap.setEnabled(spectrogram_enabled or stft_spect_enabled or 
                               spectral_centroid_enabled or ste_enabled or show_filtering_spectrograms)
        self.nfft.setEnabled(stft_enabled or spectrogram_enabled or stft_spect_enabled or 
                            spectral_centroid_enabled or ste_enabled or show_filtering_spectrograms)
        self.min_freq.setEnabled(spectrogram_enabled or stft_spect_enabled or 
                               spectral_centroid_enabled or show_filtering_spectrograms)
        self.max_freq.setEnabled(spectrogram_enabled or stft_spect_enabled or 
                               spectral_centroid_enabled or show_filtering_spectrograms)
        self.draw_style.setEnabled(spectrogram_enabled or stft_spect_enabled or 
                                 spectral_centroid_enabled or filtering_enabled)
        self.show_pitch.setEnabled(
            spectrogram_enabled or stft_spect_enabled or 
            (filtering_enabled and hasattr(self, 'waveform_radio') and not self.waveform_radio.isChecked()))

        
        # Rest of the method remains the same...
        # STE controls
        if ste_enabled and self.window_type.currentText() == 'Kaiser':
            self.beta.setEnabled(True)
        else:
            self.beta.setEnabled(False)
            
        # Pitch controls
        pitch_controls_enabled = pitch_enabled or (spectrogram_enabled and self.show_pitch.isChecked())
        self.pitch_method.setEnabled(pitch_controls_enabled)
        self.min_pitch.setEnabled(pitch_controls_enabled)
        self.max_pitch.setEnabled(pitch_controls_enabled)
        self.adv_settings.setEnabled(pitch_controls_enabled)
        
        # Filter controls
        self.filter_type.setEnabled(filtering_enabled)
        self.fund_freq.setEnabled(False)
        self.center_freq.setEnabled(False)
        self.percentage.setEnabled(False)
        self.fcut.setEnabled(False)
        self.fcut1.setEnabled(False)
        self.fcut2.setEnabled(False)

        if hasattr(self, 'waveform_radio'):  # Check if radio buttons exist
            self.waveform_radio.setEnabled(filtering_enabled)
            self.spectrogram_radio.setEnabled(filtering_enabled)
        
        if filtering_enabled:
            self.update_filter_ui(self.filter_type.currentText())
        
        self.filter_response_button.setEnabled(filtering_enabled)

    def plot_filter_response(self):
            filter_type = self.filter_type.currentText()
            percentage = float(self.percentage.text())
            
            if filter_type == 'Lowpass' or filter_type == 'Highpass':
                fcut = float(self.fcut.text())
                delta = fcut * (percentage / 100)
                
                if filter_type == 'Lowpass':
                    wp = fcut - delta
                    ws = fcut + delta
                else:
                    wp = fcut + delta
                    ws = fcut - delta
                    
                N, Wn = signal.ellipord(wp, ws, 3, 40, fs=self.fs)
                b, a = signal.ellip(N, 0.1, 40, Wn, btype=filter_type.lower(), fs=self.fs)
                
            elif filter_type == 'Harmonic':
                fund_freq = float(self.fund_freq.text())
                center_freq = float(self.center_freq.text())
                fc = fund_freq * center_freq
                fcut1 = fc - center_freq/2
                fcut2 = fc + center_freq/2
                delta1 = fcut1 * (percentage / 100)
                delta2 = fcut2 * (percentage / 100)
                
                wp1 = fcut1 + delta1
                wp2 = fcut2 - delta2
                ws1 = fcut1 - delta1
                ws2 = fcut2 + delta2
                
                N, Wn = signal.ellipord([wp1, wp2], [ws1, ws2], 3, 40, fs=self.fs)
                b, a = signal.ellip(N, 0.1, 40, Wn, btype='bandpass', fs=self.fs)
                
            else:  # Bandpass or Bandstop
                fcut1 = float(self.fcut1.text())
                fcut2 = float(self.fcut2.text())
                delta1 = fcut1 * (percentage / 100)
                delta2 = fcut2 * (percentage / 100)
                
                if filter_type == 'Bandpass':
                    wp1 = fcut1 + delta1
                    wp2 = fcut2 - delta2
                    ws1 = fcut1 - delta1
                    ws2 = fcut2 + delta2
                else:
                    wp1 = fcut1 - delta1
                    wp2 = fcut2 + delta2
                    ws1 = fcut1 + delta1
                    ws2 = fcut2 - delta2
                    
                N, Wn = signal.ellipord([wp1, wp2], [ws1, ws2], 3, 40, fs=self.fs)
                b, a = signal.ellip(N, 0.1, 40, Wn, btype=filter_type.lower(), fs=self.fs)
            
            w, h = signal.freqz(b, a, worN=8000, fs=self.fs)
            phase = np.unwrap(np.angle(h))
            
            self.current_figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            self.current_figure.suptitle(f'Filter Frequency Response ({filter_type})')
            
            ax1.plot(w, 20 * np.log10(abs(h)))
            ax1.set_title('Magnitude Response')
            ax1.set_ylabel('Amplitude [dB]')
            ax1.set_xlabel('Frequency [Hz]')
            ax1.grid(True)
            
            ax2.plot(w, phase)
            ax2.set_title('Phase Response')
            ax2.set_ylabel('Phase [radians]')
            ax2.set_xlabel('Frequency [Hz]')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()

    # [Rest of your methods remain unchanged...]
        # plot_figure(), plot_ft(), plot_stft(), plot_spectrogram(), etc.
        # All other existing methods should be kept exactly as they were in the previous implementation

    def plot_figure(self):
        method = self.method_selector.currentText()
        
        try:
            if method == 'FT':
                self.plot_ft()
            elif method == 'STFT':
                self.plot_stft()
            elif method == 'Spectrogram':
                self.plot_spectrogram()
            elif method == 'STFT + Spect':
                self.plot_stft_spect()
            elif method == 'Short-Time-Energy':
                self.plot_ste()
            elif method == 'Pitch':
                self.plot_pitch()
            elif method == 'Spectral Centroid':
                self.plot_spectral_centroid()
            elif method == 'Filtering':
                self.plot_filtering()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create plot: {str(e)}")
            if self.current_figure:
                plt.close(self.current_figure)

    def plot_ft(self):
        self.current_figure, ax = plt.subplots(2, figsize=(12,6))
        self.current_figure.suptitle('Fourier Transform')
        
        fft = np.fft.fft(self.audio) / self.lenAudio
        fft = fft[range(int(self.lenAudio/2))]
        freqs = np.arange(int(self.lenAudio/2)) / (self.lenAudio/self.fs)
        
        ax[0].plot(self.time, self.audio)
        ax[0].set(xlim=[0, self.duration], xlabel='Time (s)', ylabel='Amplitude')
        
        ax[1].plot(freqs, 20*np.log10(abs(fft)))
        ax[1].set(xlim=[0, max(freqs)], xlabel='Frequency (Hz)', ylabel='Amplitude (dB)')
        
        self.show_plot_window(self.current_figure, ax[0], self.audio)

    def plot_stft(self):
        """STFT with proper array dimension handling"""
        try:
            wind_size = float(self.window_size.text())
            nfft = int(self.nfft.currentText())

            self.current_figure, ax = plt.subplots(2, figsize=(12,6))
            self.current_figure.suptitle('STFT Analysis')

            # Trim mismatched lengths
            if len(self.time) > len(self.audio):
                self.time = self.time[:len(self.audio)]
            elif len(self.audio) > len(self.time):
                self.audio = self.audio[:len(self.time)]

            self.wind_size_samples = int(wind_size * self.fs)
            self.window = self.get_window(self.wind_size_samples)
            self.nfft_val = nfft
            self.mid_point_idx = len(self.audio) // 2

            self.update_stft_plot(ax)

            # Use unified span selector
            plot_id = id(self.current_figure.canvas.manager.window)
            self.create_span_selector(ax[0], self.audio, plot_id)

            self.current_figure.canvas.mpl_connect(
                'button_press_event',
                lambda e: self.on_window_click(e, ax)
            )

            self.show_plot_window(self.current_figure, ax[0], self.audio)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"STFT plot failed: {str(e)}")

    def on_window_click(self, event, ax):
        """Move analysis window on left click (without dragging)"""
        if event.inaxes != ax[0] or event.button != 1 or event.dblclick:
            return
        
        # Only move window on simple click (not drag)
        if hasattr(event, 'pressed') and event.pressed:
            return

        self.mid_point_idx = np.searchsorted(self.time, event.xdata)
        self.mid_point_idx = min(self.mid_point_idx, len(self.time) - 1)

            
        # Update window center position
        self.mid_point_idx = np.searchsorted(self.time, event.xdata)
        self.update_stft_plot(ax)
        
    def update_stft_plot(self, ax):
        """Update plot with proper array handling"""
        for a in ax:
            a.clear()
        
        # Current analysis window with boundary checks
        start = max(0, self.mid_point_idx - self.wind_size_samples//2)
        end = min(len(self.audio), self.mid_point_idx + self.wind_size_samples//2)
        
        # Get segments with exact matching lengths
        time_segment = self.time[start:end]
        audio_segment = self.audio[start:end]
        window_segment = self.window[:len(audio_segment)]
        
        # Ensure perfect alignment
        if len(audio_segment) < len(window_segment):
            window_segment = window_segment[:len(audio_segment)]
        elif len(window_segment) < len(audio_segment):
            audio_segment = audio_segment[:len(window_segment)]
            time_segment = time_segment[:len(window_segment)]
        
        windowed = audio_segment * window_segment
        
        # Compute STFT with padding if needed
        if len(windowed) < self.nfft_val:
            windowed = np.pad(windowed, (0, self.nfft_val - len(windowed)))
        stft = np.abs(np.fft.fft(windowed, self.nfft_val)[:self.nfft_val//2])
        freqs = np.fft.fftfreq(self.nfft_val, 1/self.fs)[:self.nfft_val//2]

        # Plotting with matched dimensions
        ax[0].plot(self.time, self.audio)
        ax[0].axvspan(time_segment[0], time_segment[-1], 
                     color='lightblue', alpha=0.3)
        ax[0].axvline(self.time[self.mid_point_idx], color='red', ls='--')
        
        ax[1].plot(freqs, 20*np.log10(stft + 1e-10))
        ax[1].set(xlim=[0, self.fs/2], xlabel='Frequency (Hz)', 
                 ylabel='Magnitude (dB)')
        
        self.current_figure.canvas.draw()

    def calculate_pitch(self, signal=None):
        """Calculate pitch using YIN algorithm with optional signal override."""
        try:
            # Use provided signal or default to self.audio
            if signal is None:
                signal = self.audio

            # Get parameters from UI
            min_pitch = float(self.min_pitch.text())
            max_pitch = float(self.max_pitch.text())
            
            # Convert to mono if stereo and normalize
            signal = np.mean(signal, axis=1) if signal.ndim > 1 else signal
            signal = librosa.util.normalize(signal)
            
            # Calculate pitch using YIN
            f0, voiced_flag, voiced_probs = librosa.pyin(
                signal,
                fmin=min_pitch,
                fmax=max_pitch,
                sr=self.fs,
                frame_length=2048,
                hop_length=512,
                fill_na=np.nan
            )
            
            # Median filter smoothing
            if len(f0) > 0:
                from scipy.ndimage import median_filter
                f0_smoothed = median_filter(f0, size=5)
                f0_smoothed[~voiced_flag] = np.nan
            else:
                f0_smoothed = f0
            
            return f0, f0_smoothed

        except Exception as e:
            QMessageBox.warning(self, "Pitch Error", f"Could not calculate pitch: {str(e)}")
            dummy_length = len(signal) // 512 if signal is not None else 100
            return np.full(dummy_length, np.nan), np.full(dummy_length, np.nan)

    def validate_spectrogram_parameters(self):
        """Validate spectrogram parameters and return calculated values"""
        # Get parameters from UI
        wind_size = float(self.window_size.text())
        overlap = float(self.overlap.text())
        nfft = int(self.nfft.currentText())
        min_freq = int(self.min_freq.text())
        max_freq = int(self.max_freq.text())
        
        # Validate frequency range
        if min_freq >= max_freq:
            raise ValueError("Minimum frequency must be less than maximum frequency")
        if min_freq < 0:
            raise ValueError("Frequency values cannot be negative")
        if max_freq > self.fs//2:
            raise ValueError(f"Maximum frequency cannot exceed Nyquist frequency ({self.fs//2} Hz)")
            
        # Validate window and overlap sizes
        if wind_size <= 0:
            raise ValueError("Window size must be positive")
        if overlap < 0:
            raise ValueError("Overlap cannot be negative")
        if overlap >= wind_size:
            raise ValueError("Overlap must be smaller than window size")
            
        # Calculate window samples and hop length with safety checks
        wind_size_samples = max(1, int(wind_size * self.fs))
        hop_size = max(1, wind_size_samples - int(overlap * self.fs))
        
        # Validate NFFT
        if nfft < wind_size_samples:
            QMessageBox.warning(self, "Warning", 
                              "NFFT should be at least as large as window size for best results")
        
        return {
            'wind_size_samples': wind_size_samples,
            'hop_size': hop_size,
            'nfft': nfft,
            'min_freq': min_freq,
            'max_freq': max_freq
        }

    def plot_spectrogram(self):
        try:
            # Validate parameters and get calculated values
            params = self.validate_spectrogram_parameters()

            wind_size_samples = params['wind_size_samples']
            hop_size = params['hop_size']
            nfft = params['nfft']
            min_freq = params['min_freq']
            max_freq = params['max_freq']
            
            # Get remaining parameters
            draw_style = self.draw_style.currentIndex() + 1
            show_pitch = self.show_pitch.isChecked()
            
            min_length = min(len(self.current_audio), len(self.time))
            audio = self.current_audio[:min_length]
            time = self.time[:min_length]
            
            fig = plt.figure(figsize=(12, 6))
            gs = plt.GridSpec(2, 2, width_ratios=[15, 1], height_ratios=[1, 3], hspace=0.1, wspace=0.05)
            ax0 = plt.subplot(gs[0, 0])
            ax1 = plt.subplot(gs[1, 0], sharex=ax0)
            cbar_ax = plt.subplot(gs[:, 1])
            fig.suptitle('Spectrogram', y=0.98)

            window = self.get_window(wind_size_samples)
            
            ax0.plot(time, audio)
            ax0.set(ylabel='Amplitude')
            
            if draw_style == 1:
                D = librosa.stft(audio, n_fft=nfft, hop_length=hop_size,
                                 win_length=wind_size_samples, window=window)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear',
                                               sr=self.fs, hop_length=hop_size, ax=ax1)

                # Manually set y-axis frequency range
                ax1.set_ylim([min_freq, max_freq])
            else:
                S = librosa.feature.melspectrogram(y=audio, sr=self.fs,
                                                   n_fft=nfft, hop_length=hop_size,
                                                   win_length=wind_size_samples,
                                                   window=window, fmin=min_freq,
                                                   fmax=max_freq)
                S_db = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                               sr=self.fs, hop_length=hop_size,
                                               fmin=min_freq, fmax=max_freq, ax=ax1)
            
            fig.colorbar(img, cax=cbar_ax, format="%+2.0f dB")
            
            #is not very accurate
            #self.annotate_brightest_frequencies(ax1, S_db, hop_size, nfft)

            if show_pitch:
                pitch, pitch_values = self.calculate_pitch()
                pitch_times = librosa.times_like(pitch_values, sr=self.fs, hop_length=hop_size)
                ax1.plot(pitch_times, pitch_values, '-', color='green', linewidth=2, alpha=0.8)

            ax0.set(xlim=[0, time[-1]])
            ax1.set(xlim=[0, time[-1]])

            self.show_plot_window(fig, ax0, audio)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Spectrogram failed: {str(e)}")
    '''
    def annotate_brightest_frequencies(self, ax, S_db, hop_length, nfft):
        """Annotate the top 3 brightest frequencies at each time frame"""
        # Find time points (every 0.5 seconds for cleaner display)
        time_points = librosa.times_like(S_db, sr=self.fs, hop_length=hop_length)
        step = max(1, int(0.5 * self.fs / hop_length))  # ~0.5 sec intervals
        
        for i in range(0, S_db.shape[1], step):
            # Get the frame's magnitude data
            frame = S_db[:, i]
            
            # Find top 3 brightest frequencies (least negative dB values)
            brightest = np.argpartition(frame, -3)[-3:]
            brightest = brightest[np.argsort(frame[brightest])][::-1]  # Sort descending
            
            for idx in brightest[:3]:  # Take top 3
                if frame[idx] > -80:  # Only show if above noise floor
                    freq = librosa.fft_frequencies(sr=self.fs, n_fft=nfft)[idx]
                    ax.annotate(f'{freq:.0f} Hz',
                               xy=(time_points[i], freq),
                               xytext=(5, 5),
                               textcoords='offset points',
                               color='white',
                               fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.2', 
                                       fc='black', alpha=0.7))
    '''

    def create_stft_plot_dialog(self, figure, waveform_ax, audio_signal):
        """Create a specialized dialog for STFT plots with live analysis button only"""
        plot_dialog = QDialog(self)
        plot_dialog.setWindowTitle("STFT Analysis")
        plot_dialog.setAttribute(Qt.WA_DeleteOnClose)
        plot_dialog.plot_id = id(plot_dialog)

        # Create main layout
        main_layout = QVBoxLayout()

        # ▶ Live analysis button (as before)
        live_btn = QPushButton("▶ Start Live Analysis")
        live_btn.setCheckable(True)
        live_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 5px;
                margin: 5px;
            }
            QPushButton:checked {
                background-color: #ffcccc;
            }
        """)
        live_btn.clicked.connect(self.toggle_live_analysis)
        main_layout.addWidget(live_btn)

        # Canvas and toolbar
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, plot_dialog)
        main_layout.addWidget(toolbar)
        main_layout.addWidget(canvas)

        # Set the final layout
        plot_dialog.setLayout(main_layout)

        # Store references
        plot_dialog.figure = figure
        plot_dialog.waveform_ax = waveform_ax
        plot_dialog.live_analysis_btn = live_btn

        # Setup span selector (as before)
        self.create_span_selector(waveform_ax, audio_signal, plot_dialog.plot_id)

        # Handle window close and cleanup
        plot_dialog.destroyed.connect(lambda: self.cleanup_plot_window(plot_dialog.plot_id))

        # Track open windows
        self.plot_windows.append(plot_dialog)

        plot_dialog.show()
        return plot_dialog


    def plot_stft_spect(self):
        """STFT + Spectrogram with interactive window selection and live analysis button"""
        try:
            wind_size = float(self.window_size.text())
            overlap = float(self.overlap.text())
            nfft = int(self.nfft.currentText())
            min_freq = int(self.min_freq.text())
            max_freq = int(self.max_freq.text())
            draw_style = self.draw_style.currentIndex() + 1
            
            # Create figure with adjusted layout
            self.current_figure = plt.figure(figsize=(12, 8))
            gs = plt.GridSpec(3, 2, width_ratios=[15, 1], height_ratios=[1, 1, 1.5], hspace=0.4)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[1, 0])
            ax3 = plt.subplot(gs[2, 0], sharex=ax1)
            cbar_ax = plt.subplot(gs[:, 1])
            
            self.current_figure.suptitle('STFT + Spectrogram', y=0.98)
            
            # Ensure time and audio arrays match
            if len(self.time) > len(self.audio):
                self.time = self.time[:len(self.audio)]
            elif len(self.audio) > len(self.time):
                self.audio = self.audio[:len(self.time)]
            
            # STFT analysis properties
            self.wind_size_samples = int(wind_size * self.fs)
            self.hop_size = self.wind_size_samples - int(overlap * self.fs)
            self.window = self.get_window(self.wind_size_samples)
            self.nfft_val = nfft

            #CHECK FOR START
            self.mid_point_idx = len(self.audio) // 2  # Start in middle
            
            # Create initial spectrogram image
            if draw_style == 1:
                D = librosa.stft(self.audio, n_fft=self.nfft_val, hop_length=self.hop_size,
                                win_length=self.wind_size_samples, window=self.window)
                self.S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                self.img = librosa.display.specshow(self.S_db, x_axis='time', y_axis='linear',
                                                sr=self.fs, hop_length=self.hop_size,
                                                ax=ax3)
                # Manually set y-axis frequency range
                ax3.set_ylim([min_freq, max_freq])
            else:
                S = librosa.feature.melspectrogram(y=self.audio, sr=self.fs,
                                                n_fft=self.nfft_val, hop_length=self.hop_size,
                                                win_length=self.wind_size_samples,
                                                window=self.window, fmin=min_freq,
                                                fmax=max_freq)
                self.S_db = librosa.power_to_db(S, ref=np.max)
                self.img = librosa.display.specshow(self.S_db, x_axis='time', y_axis='mel',
                                                sr=self.fs, hop_length=self.hop_size,
                                                fmin=min_freq, fmax=max_freq, ax=ax3)
            
            # Make spectrogram labels more readable
            ax3.set_ylabel('Frequency (Hz)', fontsize=10)
            ax3.tick_params(axis='both', which='major', labelsize=8)
            
            # Create colorbar
            self.cbar = self.current_figure.colorbar(self.img, cax=cbar_ax, format="%+2.0f dB")
            
            # Initial plot
            self.update_stft_spect_plot(ax1, ax2, ax3)
            
            plot_id = id(self.current_figure.canvas.manager.window)
            self.create_span_selector(ax1, self.audio, plot_id)
                        
            self.current_figure.canvas.mpl_connect(
                'button_press_event', 
                lambda e: self.on_window_click_spect(e, ax1, ax2, ax3)
            )
            
            # Create and show the plot dialog with live analysis button
            plot_dialog = self.create_stft_plot_dialog(self.current_figure, ax1, self.audio)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"STFT+Spectrogram plot failed: {str(e)}")

    def on_window_click_spect(self, event, ax1, ax2, ax3):
        """Move analysis window on left click (without dragging)"""
        if event.inaxes != ax1 or event.button != 1 or event.dblclick:
            return
        
        # Only move window on simple click (not drag)
        if hasattr(event, 'pressed') and event.pressed:
            return

        self.mid_point_idx = np.searchsorted(self.time, event.xdata)
        self.mid_point_idx = min(self.mid_point_idx, len(self.time) - 1)

        
        # Stop live analysis if running
        if hasattr(self, 'is_live_analysis_running') and self.is_live_analysis_running:
            self.live_analysis_timer.stop()
            self.is_live_analysis_running = False
        
        # Update window center position
        self.mid_point_idx = np.searchsorted(self.time, event.xdata)
        self.update_stft_spect_plot(ax1, ax2, ax3)

    def update_stft_spect_plot(self, ax1, ax2, ax3, segment=None):
        """Update plot with proper array handling"""
        # Clear only the plots we need to update (not the spectrogram)
        if segment is None:
            segment = self.audio  # fallback to full audio

        ax1.clear()
        ax2.clear()

        if self.mid_point_idx >= len(self.time):
            self.stop_live_analysis()
            return

        
        # Current analysis window with boundary checks
        start = max(0, self.mid_point_idx - self.wind_size_samples//2)
        end = min(len(self.audio), self.mid_point_idx + self.wind_size_samples//2)
        
        # Get segments with exact matching lengths
        time_segment = self.time[start:end]
        audio_segment = self.audio[start:end]
        window_segment = self.window[:len(audio_segment)]
        
        # Ensure perfect alignment
        if len(audio_segment) < len(window_segment):
            window_segment = window_segment[:len(audio_segment)]
        elif len(window_segment) < len(audio_segment):
            audio_segment = audio_segment[:len(window_segment)]
            time_segment = time_segment[:len(window_segment)]
        
        windowed = audio_segment * window_segment
        
        # Compute STFT with padding if needed
        if len(windowed) < self.nfft_val:
            windowed = np.pad(windowed, (0, self.nfft_val - len(windowed)))
        stft = np.abs(np.fft.fft(windowed, self.nfft_val)[:self.nfft_val//2])
        freqs = np.fft.fftfreq(self.nfft_val, 1/self.fs)[:self.nfft_val//2]
        
        # Plot time domain with highlighted window
        ax1.plot(self.time, self.audio)
        ax1.axvspan(time_segment[0], time_segment[-1], color='lightblue', alpha=0.3)
        if self.mid_point_idx < len(self.time):
            ax1.axvline(self.time[self.mid_point_idx], color='red', ls='--')
                
        # Plot STFT
        ax2.plot(freqs, 20*np.log10(stft + 1e-10))
        ax2.set(xlim=[0, self.fs/2], xlabel='Frequency (Hz)', ylabel='Magnitude (dB)')
        
        # Update the spectrogram image data instead of recreating it
        self.img.set_array(self.S_db)
        self.img.autoscale()
        
        self.current_figure.canvas.draw()

    def plot_ste(self):
        wind_size = float(self.window_size.text())
        overlap = float(self.overlap.text())
        window_type = self.window_type.currentText()
        beta = float(self.beta.text()) if window_type == 'Kaiser' else 0
        
        self.current_figure, ax = plt.subplots(2, figsize=(12,6), sharex=True)
        self.current_figure.suptitle('Short-Time Energy')
        
        # Hide x labels and tick labels for all but bottom plot
        for a in ax:
            a.label_outer()
        
        wind_size_samples = int(wind_size * self.fs)
        hop_size = wind_size_samples - int(overlap * self.fs)
        
        # Get window function
        if window_type == 'Bartlett':
            window = np.bartlett(wind_size_samples)
        elif window_type == 'Blackman':
            window = np.blackman(wind_size_samples)
        elif window_type == 'Hamming':
            window = np.hamming(wind_size_samples)
        elif window_type == 'Hanning':
            window = np.hanning(wind_size_samples)
        elif window_type == 'Kaiser':
            window = np.kaiser(wind_size_samples, beta)
        
        # Calculate STE with proper hop size and dB conversion
        ste = []
        time_points = []
        for i in range(0, len(self.audio) - wind_size_samples, hop_size):
            segment = self.audio[i:i+wind_size_samples] * window
            energy = 10 * np.log10(np.mean(segment**2))  # Convert to dB
            ste.append(energy)
            time_points.append(self.time[i + wind_size_samples//2])  # Center time point
        
        # Plot original waveform
        ax[0].plot(self.time, self.audio)
        ax[0].set(ylabel='Amplitude')
        
        # Plot STE in dB
        ax[1].plot(time_points, ste)
        ax[1].set(xlim=[0, self.duration], xlabel='Time (s)', ylabel='Amplitude (dB)')
        
        self.show_plot_window(self.current_figure, ax[0], self.audio)
    
    def plot_pitch(self):
        method = self.pitch_method.currentText()
        min_pitch = float(self.min_pitch.text())
        max_pitch = float(self.max_pitch.text())
        
        self.current_figure, ax = plt.subplots(2, figsize=(12,6))
        self.current_figure.suptitle('Pitch Contour')
        
        # No need to write to file - we can work directly with the audio array
        audio = self.audio.astype(np.float32)  # Ensure correct dtype for librosa
        
        # Plot waveform
        ax[0].plot(self.time, audio)
        
        # Calculate pitch using librosa
        if method == 'Autocorrelation':
            # Using librosa's yin algorithm (similar to autocorrelation)
            pitch_values, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=min_pitch, 
                fmax=max_pitch,
                sr=self.fs
            )
        elif method == 'Cross-correlation':
            # Using librosa's default pitch tracking (which uses STFT-based approach)
            pitch_values = librosa.yin(
                audio,
                fmin=min_pitch,
                fmax=max_pitch,
                sr=self.fs
            )
        else:
            # For other methods, you might need additional libraries
            # For example, for subharmonics you could use crepe
            raise NotImplementedError(f"Method {method} not implemented with librosa")
        
        # Create time array for pitch frames
        hop_length = 512  # default for librosa
        frame_time = librosa.frames_to_time(
            np.arange(len(pitch_values)),
            sr=self.fs,
            hop_length=hop_length
        )
        
        # Plot pitch contour
        ax[1].plot(frame_time, pitch_values, '-')
        ax[1].set(
            xlim=[0, self.duration],
            ylim=[min_pitch, max_pitch],
            xlabel='Time (s)',
            ylabel='Frequency (Hz)'
        )
        
        self.show_plot_window(self.current_figure, ax[0], self.audio)
            
    import matplotlib.gridspec as gridspec


    def plot_spectral_centroid(self):
        wind_size = float(self.window_size.text())
        overlap = float(self.overlap.text())
        nfft = int(self.nfft.currentText())
        min_freq = int(self.min_freq.text())
        max_freq = int(self.max_freq.text())
        draw_style = self.draw_style.currentIndex() + 1

        self.current_figure = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(3, 2, width_ratios=[20, 1])

        ax1 = self.current_figure.add_subplot(gs[0, 0])
        ax2 = self.current_figure.add_subplot(gs[1, 0])
        ax3 = self.current_figure.add_subplot(gs[2, 0])
        cax = self.current_figure.add_subplot(gs[2, 1])  # colorbar

        self.current_figure.suptitle('Spectral Centroid')

        wind_size_samples = int(wind_size * self.fs)
        hop_size = wind_size_samples - int(overlap * self.fs)
        window = self.get_window(wind_size_samples)

        audio_segment, (ini, end) = self.get_middle_segment(wind_size_samples)
        audio_segment = audio_segment * window

        spectral_centroid = self.calculate_sc(audio_segment)
        sc_value = f"{spectral_centroid:.2f}"

        # === Waveform ===
        ax1.plot(self.time, self.audio)
        ax1.axvspan(ini, end, color='silver', alpha=0.5)
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(self.time[0], self.time[-1])  # 👈 this is the fix


        # === PSD ===
        _, freqs = ax2.psd(audio_segment, NFFT=wind_size_samples, Fs=self.fs,
                           window=window, noverlap=int(overlap * self.fs))
        ax2.axvline(x=spectral_centroid, color='r')
        ax2.set_xlim([0, max(freqs)])
        ax2.set_ylabel("Power")
        ax2.set_title(f"Spectral Centroid: {sc_value} Hz")

        # === Spectrogram ===
        if draw_style == 1:
            D = librosa.stft(self.audio, n_fft=nfft, hop_length=hop_size,
                             win_length=wind_size_samples, window=window)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear',
                                           sr=self.fs, hop_length=hop_size,
                                           fmin=min_freq, fmax=max_freq, ax=ax3)
            ax3.set_ylim([min_freq, max_freq])
        else:
            S = librosa.feature.melspectrogram(y=self.audio, sr=self.fs,
                                               n_fft=nfft, hop_length=hop_size,
                                               win_length=wind_size_samples,
                                               window=window, fmin=min_freq,
                                               fmax=max_freq)
            S_db = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                           sr=self.fs, hop_length=hop_size,
                                           fmin=min_freq, fmax=max_freq, ax=ax3)

        # === Overlay spectral centroid on spectrogram ===
        sc = librosa.feature.spectral_centroid(y=self.audio, sr=self.fs,
                                               n_fft=nfft, hop_length=hop_size,
                                               win_length=wind_size_samples)
        times = librosa.times_like(sc, sr=self.fs, hop_length=hop_size)
        ax3.plot(times, sc[0], color='w', linewidth=1.5)
        ax3.set_ylabel("Freq (Hz)")
        ax3.set_xlabel("Time (s)")

        self.current_figure.colorbar(img, cax=cax, format="%+2.0f dB")
        plt.tight_layout(rect=[0, 0, 0.97, 0.95])  # leave space for title

        self.show_plot_window(self.current_figure, ax1, self.audio)



    def plot_filtered_waveform(self, filter_type, filtered_signal):
        """Plot original and filtered waveforms with proper span selectors"""
        self.current_figure, ax = plt.subplots(2, figsize=(12,6))
        self.current_figure.suptitle(f'Filtered Signal ({filter_type}) - Waveform')
        
        # Plot original signal
        ax[0].plot(self.time, self.audio)
        ax[0].set(xlim=[0, self.duration], title='Original Signal')
        
        # Plot filtered signal
        ax[1].plot(self.time, filtered_signal)
        ax[1].set(xlim=[0, self.duration], title='Filtered Signal')
        
        # Get unique identifier for this plot window
        plot_id = id(self.current_figure.canvas.manager.window)
        
        # Create span selectors for both signals
        self.create_span_selector(ax[0], self.audio, plot_id)  # Note: ax[0] not ax0
        self.create_span_selector(ax[1], filtered_signal, plot_id)  # Note: ax[1] not ax1
        
        self.show_plot_window(self.current_figure, ax[0], self.audio)


    #similar to createSpanSelector just this takes the audio
    def create_span_selector(self, ax, audio_signal, plot_id):
        """Create a span selector for a specific axis within a plot window"""

        def onselect(xmin, xmax):
            self.stop_all_audio()

            # Convert time to sample indices
            start_sample = int(xmin * self.fs)
            end_sample = int(xmax * self.fs)

            # Safety check
            if start_sample >= end_sample or end_sample > len(audio_signal):
                print("Invalid selection range.")
                return

            segment = audio_signal[start_sample:end_sample].astype(np.float32)
            self.stream_pos = 0
            self.stream_data = segment
            total_length = len(segment)

            def callback(outdata, frames, time, status):
                if status:
                    print("Stream status:", status)

                remaining = total_length - self.stream_pos
                if remaining <= 0:
                    raise sd.CallbackStop()

                n_frames = min(frames, remaining)
                chunk = self.stream_data[self.stream_pos:self.stream_pos + n_frames]

                if n_frames < frames:
                    chunk = np.pad(chunk, (0, frames - n_frames))

                outdata[:, 0] = chunk
                self.stream_pos += n_frames

                if self.stream_pos >= total_length:
                    raise sd.CallbackStop()

            try:
                self.active_stream = sd.OutputStream(
                    samplerate=self.fs,
                    channels=1,
                    dtype='float32',
                    callback=callback,
                    blocksize=4096,
                )
                self.active_stream.start()
            except Exception as e:
                print("Error starting audio stream:", e)

            # Visual: remove old span if any and draw new
            try:
                for patch in list(ax.patches):
                    if hasattr(patch, 'get_label') and patch.get_label() == 'selection':
                        patch.remove()
            except Exception:
                pass

            ax.axvspan(xmin, xmax, color='red', alpha=0.3, label='selection')
            ax.figure.canvas.draw_idle()

        # Ensure this plot ID has a selector list
        if plot_id not in self.span_selectors:
            self.span_selectors[plot_id] = []

        # Disconnect any existing selector on this axis
        for existing_selector in self.span_selectors[plot_id]:
            if getattr(existing_selector, 'ax', None) == ax:
                try:
                    existing_selector.disconnect_events()
                except Exception:
                    pass

        selector = SpanSelector(
            ax,
            onselect,
            'horizontal',
            useblit=True,
            interactive=True,
            drag_from_anywhere=True
        )
        selector.ax = ax
        self.span_selectors[plot_id].append(selector)




    def plot_filtered_spectrogram(self, filter_type, filtered_signal):
        """Plot original and filtered spectrograms with optional pitch curves."""
        try:
            # Validate parameters and get calculated values
            params = self.validate_spectrogram_parameters()
            wind_size_samples = params['wind_size_samples']
            hop_size = params['hop_size']
            nfft = params['nfft']
            min_freq = params['min_freq']
            max_freq = params['max_freq']

            show_pitch = self.show_pitch.isChecked()
            draw_style = self.draw_style.currentIndex() + 1
            window = self.get_window(wind_size_samples)

            self.current_figure = plt.figure(figsize=(12, 8))
            gs = plt.GridSpec(2, 1, height_ratios=[1, 1])
            plot_id = id(self.current_figure.canvas.manager.window)

            # ---- ORIGINAL SPECTROGRAM ----
            ax0 = plt.subplot(gs[0])
            img0 = None
            if draw_style == 1:
                D_orig = librosa.stft(self.audio, n_fft=nfft, hop_length=hop_size,
                                      win_length=wind_size_samples, window=window)
                S_db_orig = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)
                

                img0 = librosa.display.specshow(S_db_orig, x_axis='time', y_axis='linear',
                                                sr=self.fs, hop_length=hop_size,
                                                fmin=min_freq, fmax=max_freq, ax=ax0)
                # Manually set y-axis frequency range
                ax0.set_ylim([min_freq, max_freq])
            else:
                S_orig = librosa.feature.melspectrogram(y=self.audio, sr=self.fs,
                                                        n_fft=nfft, hop_length=hop_size,
                                                        win_length=wind_size_samples,
                                                        window=window, fmin=min_freq,
                                                        fmax=max_freq)
                S_db_orig = librosa.power_to_db(S_orig, ref=np.max)
                img0 = librosa.display.specshow(S_db_orig, x_axis='time', y_axis='mel',
                                                sr=self.fs, hop_length=hop_size,
                                                fmin=min_freq, fmax=max_freq, ax=ax0)
            # correction  for the lengths
            duration = librosa.frames_to_time(S_db_orig.shape[1], sr=self.fs, hop_length=hop_size)
            ax0.set_xlim([0, duration])
            

            ax0.set(title='Original Signal Spectrogram')
            #not work
            print(f"Creating span selector for ax0 (original) - plot_id={plot_id}")
            self.create_span_selector(ax0, self.audio, plot_id)

            if show_pitch:
                _, pitch_smoothed = self.calculate_pitch(signal=self.audio)
                pitch_times = librosa.times_like(pitch_smoothed, sr=self.fs, hop_length=hop_size)
                ax0.plot(pitch_times, pitch_smoothed, '-', color='lime', linewidth=2, alpha=0.9, label='Pitch')


            # ---- FILTERED SPECTROGRAM ----
            ax1 = plt.subplot(gs[1])
            img1 = None
            if draw_style == 1:
                D_filt = librosa.stft(filtered_signal, n_fft=nfft, hop_length=hop_size,
                                      win_length=wind_size_samples, window=window)
                S_db_filt = librosa.amplitude_to_db(np.abs(D_filt), ref=np.max)
                img1 = librosa.display.specshow(S_db_filt, x_axis='time', y_axis='linear',
                                                sr=self.fs, hop_length=hop_size,
                                                ax=ax1)

                # Manually set y-axis frequency range
                ax1.set_ylim([min_freq, max_freq])
            else:
                S_filt = librosa.feature.melspectrogram(y=filtered_signal, sr=self.fs,
                                                        n_fft=nfft, hop_length=hop_size,
                                                        win_length=wind_size_samples,
                                                        window=window, fmin=min_freq,
                                                        fmax=max_freq)
                S_db_filt = librosa.power_to_db(S_filt, ref=np.max)
                img1 = librosa.display.specshow(S_db_filt, x_axis='time', y_axis='mel',
                                                sr=self.fs, hop_length=hop_size,
                                                fmin=min_freq, fmax=max_freq, ax=ax1)
            duration_filt = librosa.frames_to_time(S_db_filt.shape[1], sr=self.fs, hop_length=hop_size)
            ax1.set_xlim([0, duration_filt])

            ax1.set(title=f'Filtered Signal Spectrogram ({filter_type})')
            
            print(f"Creating span selector for ax1 (filtered) - plot_id={plot_id}")
            self.create_span_selector(ax1, filtered_signal, plot_id)

            if show_pitch:
                _, pitch_smoothed_filt = self.calculate_pitch(signal=filtered_signal)
                pitch_times_filt = librosa.times_like(pitch_smoothed_filt, sr=self.fs, hop_length=hop_size)
                ax1.plot(pitch_times_filt, pitch_smoothed_filt, '-', color='cyan', linewidth=2, alpha=0.9, label='Pitch')
                 #ax1.legend(loc='upper right')
            '''
            # ---- COLORBARS ----
            if img0 is not None and img1 is not None:
                self.current_figure.subplots_adjust(right=0.85)
                cbar_ax0 = self.current_figure.add_axes([0.88, 0.55, 0.02, 0.35])
                self.current_figure.colorbar(img0, cax=cbar_ax0, format="%+2.0f dB")
                cbar_ax0.set_ylabel('Original Signal')

                cbar_ax1 = self.current_figure.add_axes([0.88, 0.15, 0.02, 0.35])
                self.current_figure.colorbar(img1, cax=cbar_ax1, format="%+2.0f dB")
                cbar_ax1.set_ylabel('Filtered Signal')
            '''

            #first change
            self.current_figure.tight_layout()
            self.show_plot_window(self.current_figure, None, None, create_selector=False)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Spectrogram failed: {str(e)}")
            if self.current_figure:
                plt.close(self.current_figure)


    def plot_filtering(self):
        filter_type = self.filter_type.currentText()
        percentage = float(self.percentage.text())
        
        if filter_type == 'Lowpass' or filter_type == 'Highpass':
            fcut = float(self.fcut.text())
            delta = fcut * (percentage / 100)
            
            if filter_type == 'Lowpass':
                wp = fcut - delta
                ws = fcut + delta
            else:
                wp = fcut + delta
                ws = fcut - delta
                
            N, Wn = signal.ellipord(wp, ws, 3, 40, fs=self.fs)
            b, a = signal.ellip(N, 0.1, 40, Wn, btype=filter_type.lower(), fs=self.fs)
            
        elif filter_type == 'Harmonic':
            fund_freq = float(self.fund_freq.text())
            center_freq = float(self.center_freq.text())
            fc = fund_freq * center_freq
            fcut1 = fc - center_freq/2
            fcut2 = fc + center_freq/2
            delta1 = fcut1 * (percentage / 100)
            delta2 = fcut2 * (percentage / 100)
            
            wp1 = fcut1 + delta1
            wp2 = fcut2 - delta2
            ws1 = fcut1 - delta1
            ws2 = fcut2 + delta2
            
            N, Wn = signal.ellipord([wp1, wp2], [ws1, ws2], 3, 40, fs=self.fs)
            b, a = signal.ellip(N, 0.1, 40, Wn, btype='bandpass', fs=self.fs)
            
        else:  # Bandpass or Bandstop
            fcut1 = float(self.fcut1.text())
            fcut2 = float(self.fcut2.text())
            delta1 = fcut1 * (percentage / 100)
            delta2 = fcut2 * (percentage / 100)
            
            if filter_type == 'Bandpass':
                wp1 = fcut1 + delta1
                wp2 = fcut2 - delta2
                ws1 = fcut1 - delta1
                ws2 = fcut2 + delta2
            else:
                wp1 = fcut1 - delta1
                wp2 = fcut2 + delta2
                ws1 = fcut1 + delta1
                ws2 = fcut2 - delta2
                
            N, Wn = signal.ellipord([wp1, wp2], [ws1, ws2], 3, 40, fs=self.fs)
            b, a = signal.ellip(N, 0.1, 40, Wn, btype=filter_type.lower(), fs=self.fs)
        
        filtered_signal = signal.lfilter(b, a, self.audio)
        
        self.current_figure, ax = plt.subplots(2, figsize=(12,6))
        self.current_figure.suptitle(f'Filtered Signal ({filter_type})')
        
        if self.waveform_radio.isChecked():
            self.plot_filtered_waveform(filter_type, filtered_signal)
        else:
            self.plot_filtered_spectrogram(filter_type, filtered_signal)

    def calculate_sc(self, segment):
        magnitudes = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), 1/self.fs)
        return np.sum(magnitudes * freqs) / np.sum(magnitudes)

    def cleanup_plot_window(self, plot_id):
        """Clean up when any plot window closes"""
        # Stop live analysis if active
        if hasattr(self, 'live_timer') and self.live_timer.isActive():
            self.stop_live_analysis()
        
        # Clean up span selectors
        if plot_id in self.span_selectors:
            for selector in self.span_selectors[plot_id]:
                try:
                    selector.disconnect_events()
                except:
                    pass
            del self.span_selectors[plot_id]
        
        # Remove window reference
        self.plot_windows = [w for w in self.plot_windows if getattr(w, 'plot_id', None) != plot_id]
        
        # Clear current figure if it's the one being closed
        if hasattr(self, 'current_figure') and id(self.current_figure) == plot_id:
            del self.current_figure

    def show_plot_window(self, figure, waveform_ax=None, audio_signal=None, create_selector=True):
        """Show plot window with proper close handling"""
        method = self.method_selector.currentText()
        start, end = 0, self.duration
        plot_title = f"{method}_{self.base_name}_{self.format_timestamp(start)}-{self.format_timestamp(end)}"

        plot_dialog = QDialog(self)
        plot_dialog.setWindowTitle(plot_title)
        plot_dialog.setAttribute(Qt.WA_DeleteOnClose)
        plot_dialog.plot_id = id(plot_dialog)

        # Store references
        plot_dialog.figure = figure
        plot_dialog.waveform_ax = waveform_ax
        self.current_figure = figure

        layout = QVBoxLayout()
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, plot_dialog)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        plot_dialog.setLayout(layout)

        def handle_close():
            self.on_plot_window_close(plot_dialog.plot_id)
        plot_dialog.finished.connect(handle_close)

        # Only create selector if valid
        if create_selector and waveform_ax is not None and audio_signal is not None:
            self.create_span_selector(waveform_ax, audio_signal, plot_dialog.plot_id)

        self.plot_windows.append(plot_dialog)
        plot_dialog.show()
        return plot_dialog


    def on_plot_window_close(self, plot_id):
        """Handle plot window closure"""
        # Stop live analysis if active
        if hasattr(self, 'live_timer') and self.live_timer.isActive():
            self.stop_live_analysis()
        
        # Clean up span selectors
        if plot_id in self.span_selectors:
            for selector in self.span_selectors[plot_id]:
                try:
                    selector.disconnect_events()
                except:
                    pass
            del self.span_selectors[plot_id]
        
        # Remove window reference
        self.plot_windows = [w for w in self.plot_windows if getattr(w, 'plot_id', None) != plot_id]
        
        # Clear current figure if it's the one being closed
        if hasattr(self, 'current_figure') and id(self.current_figure) == plot_id:
            del self.current_figure

    def get_window(self, size):
        window_type = self.window_type.currentText()
        
        if window_type == 'Bartlett':
            return np.bartlett(size)
        elif window_type == 'Blackman':
            return np.blackman(size)
        elif window_type == 'Hamming':
            return np.hamming(size)
        elif window_type == 'Hanning':
            return np.hanning(size)
        elif window_type == 'Kaiser':
            return np.kaiser(size, float(self.beta.text()))
            
    def get_middle_segment(self, window_size):
        """Get middle segment of audio with proper size handling"""
        mid = len(self.audio) // 2
        start = mid - window_size // 2
        end = mid + window_size // 2
        
        if start < 0:
            start = 0
        if end > len(self.audio):
            end = len(self.audio)
        
        segment = self.audio[start:end]
        time_range = (self.time[start], self.time[end-1])  # Note the -1 to avoid index overflow
        
        # Pad with zeros if needed to match window size
        if len(segment) < window_size:
            pad_size = window_size - len(segment)
            segment = np.pad(segment, (0, pad_size), 'constant')
        
        return segment, time_range

    def closeEvent(self, event):
        # Close all plot windows
        for window in self.plot_windows[:]:  # Iterate over copy
            try:
                window.close()
            except RuntimeError:
                pass  # Already closed
        
        # Clear the list
        self.plot_windows.clear()

        if hasattr(self, 'live_analysis_timer'):
            self.live_analysis_timer.stop()
        
        # Close figures
        if hasattr(self, 'current_figure') and self.current_figure:
            plt.close(self.current_figure)
        
        event.accept()
        super().closeEvent(event)





