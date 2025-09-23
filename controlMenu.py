
#douleyei me ta ola prin to start/pause.

from PyQt5.QtWidgets import (QFileDialog, QWidget, QDoubleSpinBox, QSpinBox, QMessageBox, QSlider, QHBoxLayout, QDialog, QLabel, QPushButton, QLineEdit, QRadioButton, 
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
import matplotlib.gridspec as gridspec
from matplotlib.widgets import SpanSelector
from matplotlib import patches


import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import os


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
        self.controller = controller
        
        # CRITICAL SETTINGS:
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
            'Waveform', 'Fourier Transform', 'Short Time Fourier Transform', 'Spectrogram', 'STFT + Spect', 
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
     
        self.help_button = QPushButton('🛈 Help')
        self.help_button.setFixedWidth(150)
        self.help_button.clicked.connect(self.show_help)
        main_layout.addWidget(self.help_button, 14, 2, 1, 1, Qt.AlignRight)

        # Font size controls
        font_layout = QHBoxLayout()
        font_label = QLabel("Font Size(in plots):")
        self.font_spin = QSpinBox()
        self.font_spin.setRange(8, 24)
        self.font_spin.setValue(12)
        self.font_spin.valueChanged.connect(self.update_font_setting)
        font_layout.addWidget(font_label)
        font_layout.addWidget(self.font_spin)

        # Add save button next to plot button
        self.save_button = QPushButton('Save to Excel')
        self.save_button.clicked.connect(self.save_to_excel)
        main_layout.addWidget(self.save_button, 14, 0, 1, 1)
        
        # Add font controls to the right of help button
        font_container = QWidget()
        font_container.setLayout(font_layout)
        main_layout.addWidget(font_container, 14, 1, 1, 1, Qt.AlignRight)
        
        self.setLayout(main_layout)
        self.update_ui_state('Spectrogram')

    def update_font_setting(self, size):
        """Update the global font setting"""
        self.current_font_size = size

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

    def save_to_excel(self):
        from PyQt5.QtWidgets import QFileDialog
        import pandas as pd

        # Open save dialog
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Audio Data to Excel",
            f"{self.base_name}.xlsx",
            "Excel Files (*.xlsx)",
            options=options
        )
        if not file_path:
            return  # User cancelled

        print("len(self.time) =", len(self.time))
        print("len(self.audio) =", len(self.audio))

        # Ensure equal length
        min_len = min(len(self.time), len(self.audio))
        t = self.time[:min_len]
        y = self.audio[:min_len]

        # Build DataFrame
        df = pd.DataFrame({
            "Time (s)": t,
            "Amplitude": y
        })

        # Save as CSV instead of Excel
        csv_path = file_path.replace(".xlsx", ".csv")
        df.to_csv(csv_path, index=False)

        print(f"Saved {len(df)} rows to {csv_path}")

    # Audio helpers.
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
            self.stop_live_analysis()  # Ensure everything shuts down
            if hasattr(self, 'live_analysis_button'):
                self.live_analysis_button.setText("▶ Play Audio")

        print(f"Updating position: mid_point_idx = {self.mid_point_idx}")

    def start_audio_playback(self):
        """Start audio playback from current cursor position"""
        self.stop_audio_playback()  # Ensure previous playback is stopped

        self.mid_point_idx = 0 
        start_sample = int(self.mid_point_idx)
        audio_segment = self.audio[start_sample:]

        if np.max(np.abs(audio_segment)) > 1.0:
            audio_segment = audio_segment / np.max(np.abs(audio_segment))

        self.is_playing = True
        self.playback_start_time = time.time()
        self.playback_start_sample = start_sample

        import sounddevice as sd
        sd.stop()
        self.current_stream = sd.play(audio_segment, self.fs, blocking=False)

        if not hasattr(self, 'playback_timer'):
            self.playback_timer = QTimer()
            self.playback_timer.timeout.connect(self.update_playback_position)
        self.playback_timer.start(30)  # ~30 FPS

    def stop_audio_playback(self):
        """Stop any ongoing audio playback"""
        import sounddevice as sd
        sd.stop()
        self.is_playing_audio = False
        self.current_stream = None
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
        self.stop_audio_playback()  # Ensure audio stops

        if hasattr(self, 'live_timer'):
            self.live_timer.stop()

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
        except (AttributeError, RuntimeEr,ror) as e:
            print(f"Visual update failed: {e}")
            self.stop_live_analysis()
        
        # Auto-stop at end
        if self.mid_point_idx >= len(self.audio) - self.wind_size_samples//2:
            self.stop_live_analysis()

    # GUI settting up.

    def create_pitch_group(self, layout):
        group = QGroupBox("Pitch")
        grid = QGridLayout()
        
        self.pitch_method = QComboBox()
        self.pitch_method.addItems(['Autocorrelation', 'Cross-correlation', 'Subharmonics', 'Spinet'])
        
        self.min_pitch = QLineEdit("75.0")
        self.max_pitch = QLineEdit("600.0")
        
        grid.addWidget(QLabel("Method:"), 0, 0)
        grid.addWidget(self.pitch_method, 0, 1)
        grid.addWidget(QLabel("Min pitch (Hz):"), 1, 0)
        grid.addWidget(self.min_pitch, 1, 1)
        grid.addWidget(QLabel("Max pitch (Hz):"), 2, 0)
        grid.addWidget(self.max_pitch, 2, 1)
        
        group.setLayout(grid)
        layout.addWidget(group, 9, 0, 5, 2)

    def toggle_pitch_controls(self, state):
        enabled = state == Qt.Checked
        self.pitch_method.setEnabled(enabled)
        self.min_pitch.setEnabled(enabled)
        self.max_pitch.setEnabled(enabled)

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

        self.window_type.currentTextChanged.connect(self.update_beta_state)
    
    def update_beta_state(self):
        """Update beta field enabled state based on window selection"""
        if self.window_type.currentText() == 'Kaiser':
            self.beta.setEnabled(True)
        else:
            self.beta.setEnabled(False)

    def update_ui_state(self, method):
        # First define all the mode flags

        wave_enabled = method == 'Waveform'
        ft_enabled = method == 'Fourier Transform'
        stft_enabled = method == 'Short Time Fourier Transform'
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
        self.min_freq.setEnabled(ft_enabled or stft_enabled or spectrogram_enabled or stft_spect_enabled or 
                         spectral_centroid_enabled or show_filtering_spectrograms)
        self.max_freq.setEnabled(ft_enabled or stft_enabled or spectrogram_enabled or stft_spect_enabled or 
                                 spectral_centroid_enabled or show_filtering_spectrograms)
        self.draw_style.setEnabled(spectrogram_enabled or stft_spect_enabled or 
                                 spectral_centroid_enabled or filtering_enabled)
        self.show_pitch.setEnabled(
            spectrogram_enabled or stft_spect_enabled or ste_enabled or
            (filtering_enabled and hasattr(self, 'waveform_radio') and not self.waveform_radio.isChecked()))

        
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

    def get_freq_bounds(self):
        try:
            min_freq = float(self.min_freq.text())
        except Exception:
            min_freq = 0.0

        try:
            max_freq = float(self.max_freq.text())
        except Exception:
            max_freq = self.fs / 2

        return min_freq, max_freq

    ### PLOTS ###

    def plot_figure(self):
        method = self.method_selector.currentText()
        
        try:
            if method == 'Fourier Transform':
                self.plot_ft()
            elif method == 'Waveform':
                self.plot_waveform()
            elif method == 'Short Time Fourier Transform':
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

    # WAVEFORM
    def plot_waveform(self):
        # Get current font size (default to 12 if not set)
        fontsize = getattr(self, 'current_font_size', 12)
        
        # Set style before creating figure
        plt.style.use('default')
        plt.rcParams.update({'font.size': fontsize})

        self.current_figure, ax = plt.subplots(figsize=(12, 6))
        self.current_figure.suptitle('Waveform')

        ax.plot(self.time, self.audio)
        ax.set(xlim=[0, self.duration], xlabel='Time (s)', ylabel='Amplitude')
        ax.tick_params(axis='both', labelsize=fontsize*0.9)  # Slightly smaller ticks
            

        def format_time_amp(x, y):
            return f"time = {x:.2f} s, amplitude = {y:.3f}"

        ax.format_coord = format_time_amp   # time-domain waveform

        self.show_plot_window(self.current_figure, ax, self.audio)
        
    # FT
    def plot_ft(self):
        # Get current font size (default to 12 if not set)
        fontsize = getattr(self, 'current_font_size', 12)
        
        # Set style before creating figure
        plt.style.use('default')
        plt.rcParams.update({'font.size': fontsize})

        self.current_figure, ax = plt.subplots(2, figsize=(12,6))
        self.current_figure.suptitle('Fourier Transform')

        fft = np.fft.fft(self.audio) / self.lenAudio
        fft = fft[range(int(self.lenAudio/2))]
        freqs = np.arange(int(self.lenAudio/2)) / (self.lenAudio/self.fs)

        ax[0].plot(self.time, self.audio)
        ax[0].set(xlim=[0, self.duration], xlabel='Time (s)', ylabel='Amplitude')
        ax[0].tick_params(axis='both', labelsize=fontsize*0.9)  # Slightly smaller ticks

        min_freq, max_freq = self.get_freq_bounds()
        
        # Calculate magnitude and convert to dB (relative scale)
        magnitude = np.abs(fft)
        # Use a small epsilon to avoid log(0) which would be -infinity
        epsilon = 1e-10
        magnitude_db = 20 * np.log10(magnitude + epsilon)
        
        # Normalize to have 0 dB as the maximum (optional but common)
        # magnitude_db = 20 * np.log10(magnitude/np.max(magnitude) + epsilon)
        
        def format_time_amp(x, y):
            return f"time = {x:.2f} s, amplitude = {y:.3f}"

        def format_freq_db(x, y):
            return f"freq = {x:.1f} Hz, magnitude = {y:.1f} dB"

        ax[0].format_coord = format_time_amp   # time-domain waveform
        ax[1].format_coord = format_freq_db    # FFT window

        ax[1].plot(freqs, magnitude_db)
        ax[1].set(xlim=[min_freq, max_freq], xlabel='Frequency (Hz)', ylabel='Magnitude (dB)')

        self.show_plot_window(self.current_figure, ax[0], self.audio)

    # STFT
    def plot_stft(self):
        """STFT with proper array dimension handling"""
        try:
            # Get current font size (default to 12 if not set)
            fontsize = getattr(self, 'current_font_size', 12)
            
            # Set style before creating figure
            plt.style.use('default')
            plt.rcParams.update({'font.size': fontsize})

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
            #plot_id = id(self.current_figure.canvas.manager.window)
            #self.create_span_selector(ax[0], self.audio, plot_id)

            self.current_figure.canvas.mpl_connect(
                'button_press_event',
                lambda e: self.on_window_click(e, ax)
            )

            self.show_plot_window(self.current_figure, ax[0], self.audio)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"STFT plot failed: {str(e)}")

    def on_window_click(self, event, ax):
        """Handle ONLY simple clicks for window movement"""
        if event.inaxes != ax[0] or event.button != 1:
            return
            
        # Skip if this is part of a drag operation
        if hasattr(event, 'pressed') and event.pressed:
            return
            
        # Skip if we're currently dragging a span selector
        if hasattr(self, '_is_dragging_span') and self._is_dragging_span:
            return
            
        # Move analysis window to click position
        self.mid_point_idx = np.searchsorted(self.time, event.xdata)
        self.mid_point_idx = min(self.mid_point_idx, len(self.time) - 1)
        self.update_stft_plot(ax)

    def update_stft_plot(self, ax):
        """Update plot with proper array handling"""
        for a in ax[:2]:  # Clear only the first two axes
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

        # Normalize the STFT similar to how you normalize the FT
        stft = np.fft.fft(windowed, self.nfft_val) / len(windowed)
        stft = stft[:self.nfft_val//2]  # Take only positive frequencies
        freqs = np.fft.fftfreq(self.nfft_val, 1/self.fs)[:self.nfft_val//2]

        # Plotting with matched dimensions
        ax[0].plot(self.time, self.audio)
        ax[0].set_xlim(self.time[0], self.time[-1])
        ax[0].axvspan(time_segment[0], time_segment[-1], 
                     color='lightblue', alpha=0.3)
        ax[0].axvline(self.time[self.mid_point_idx], color='red', ls='--')
        ax[0].set_ylabel('Amplitude')
        ax[0].set_title('Time Domain Signal')
        
        # Calculate magnitude and convert to dB (normalized)
        magnitude = np.abs(stft)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        min_freq, max_freq = self.get_freq_bounds()
        ax[1].plot(freqs, magnitude_db)
        ax[1].set(xlim=[min_freq, max_freq], xlabel='Frequency (Hz)', 
                 ylabel='Magnitude (dB)')
        ax[1].set_title('Frequency Spectrum at Selected Window')
        
        def format_time_amp(x, y):
            return f"time = {x:.2f} s, amplitude = {y:.3f}"

        def format_freq_db(x, y):
            return f"freq = {x:.1f} Hz, magnitude = {y:.1f} dB"

        ax[0].format_coord = format_time_amp   # time-domain waveform
        ax[1].format_coord = format_freq_db    # FFT window

        self.current_figure.canvas.draw()

    # Pitch
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

    def plot_pitch(self):
        # Get current font size (default to 12 if not set)
        fontsize = getattr(self, 'current_font_size', 12)
        
        # Set style before creating figure
        plt.style.use('default')
        plt.rcParams.update({'font.size': fontsize})


        method = self.pitch_method.currentText()
        min_pitch = float(self.min_pitch.text())
        max_pitch = float(self.max_pitch.text())
        
        self.current_figure, ax = plt.subplots(2, figsize=(12,6))
        self.current_figure.suptitle('Pitch Contour')
        
        audio = self.audio.astype(np.float32)  # Ensure correct dtype for librosa
        

        duration = len(audio) / self.fs  # This is the correct way
        # Plot waveform with proper xlim
        ax[0].plot(self.time, audio)
        ax[0].set_xlim([0, duration])  # Set x-axis limits based on actual duration
        ax[0].set_title('Waveform')


        
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

        def format_time_amp(x, y):
            return f"time = {x:.2f} s, amplitude = {y:.3f}"
        def format_time_freq(x, y):
            return f"time = {x:.2f} s, freq = {y:.1f} Hz"
                
        ax[0].format_coord = format_time_amp   # time-domain waveform
        ax[1].format_coord = format_time_freq  # spectrogram
        
        self.show_plot_window(self.current_figure, ax[0], self.audio)

    # Spectrogram
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
            # Get current font size (default to 12 if not set)
            fontsize = getattr(self, 'current_font_size', 12)
            
            # Set style before creating figure
            plt.style.use('default')
            plt.rcParams.update({'font.size': fontsize})

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
            
            if show_pitch:
                pitch, pitch_values = self.calculate_pitch()
                pitch_times = librosa.times_like(pitch_values, sr=self.fs, hop_length=hop_size)
                ax1.plot(pitch_times, pitch_values, '-', color='green', linewidth=2, alpha=0.8)

            ax0.set(xlim=[0, time[-1]])
            ax1.set(xlim=[0, time[-1]])

            def format_time_amp(x, y):
                return f"time = {x:.2f} s, amplitude = {y:.3f}"
            def format_time_freq(x, y):
                return f"time = {x:.2f} s, freq = {y:.1f} Hz"

            ax0.format_coord = format_time_amp   # time-domain waveform
            ax1.format_coord = format_time_freq  # spectrogram


            # Store both axes in the plot dialog by passing them as a tuple
            plot_dialog = self.show_plot_window(fig, ax0, audio)
            
            # Store the spectrogram axis separately for easy access
            plot_dialog.spectrogram_ax = ax1
            return plot_dialog
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Spectrogram failed: {str(e)}")

    # STFT + Spectro
    def update_live_analysis(self, dialog):
        """Update the analysis window position for live analysis synchronized with audio"""
        # Safety check: ensure dialog still exists
        try:
            # This will raise a RuntimeError if the dialog has been deleted
            _ = dialog.objectName()
        except RuntimeError:
            # Dialog has been deleted, stop the timer if it exists
            if hasattr(dialog, 'live_analysis_timer') and dialog.live_analysis_timer is not None:
                try:
                    dialog.live_analysis_timer.stop()
                except:
                    pass
            return
        
        if not hasattr(self, 'is_live_analysis_running') or not self.is_live_analysis_running:
            return
        
        # If audio playback is enabled, try to synchronize with the audio position
        audio_playing = False
        if (hasattr(dialog, 'audio_playback_checkbox') and 
            dialog.audio_playback_checkbox.isChecked() and
            hasattr(dialog, 'audio_start_time')):
            
            try:
                # Calculate the current position based on elapsed time
                elapsed_time = time.time() - dialog.audio_start_time
                current_sample = int(elapsed_time * self.fs)
                audio_playing = True
                
                # Check if we've reached the end
                if current_sample >= len(self.audio):
                    # Stop at the end
                    self.toggle_live_analysis(False, dialog)
                    return
                
                self.mid_point_idx = current_sample
            except Exception as e:
                print(f"Error in audio synchronization: {e}")
                # Audio playback not working, fall back to timer-based movement
                audio_playing = False
        
        # If audio is not playing or not enabled, use timer-based movement
        # BUT only if audio playback is not enabled
        if not audio_playing and not (hasattr(dialog, 'audio_playback_checkbox') and dialog.audio_playback_checkbox.isChecked()):
            step_size = self.hop_size
            new_mid_point = self.mid_point_idx + step_size
            
            # Check if we've reached the end
            if new_mid_point >= len(self.audio):
                # Stop at the end instead of wrapping around
                self.toggle_live_analysis(False, dialog)
                return
            
            self.mid_point_idx = new_mid_point
        
        # Update the plots
        if hasattr(self, 'current_figure') and self.current_figure:
            # Find the axes from the current figure
            axes = self.current_figure.get_axes()
            if len(axes) >= 3:
                self.update_stft_spect_plot(axes[0], axes[1], axes[2])

    def toggle_live_analysis(self, checked, dialog=None):
        """Toggle live analysis scrolling with audio playback"""
        # If no dialog is passed, try to get it from the current figure
        if dialog is None and hasattr(self, 'current_figure'):
            for window in self.plot_windows:
                if hasattr(window, 'figure') and window.figure == self.current_figure:
                    dialog = window
                    break
        
        if dialog is None:
            print("No dialog found for live analysis")
            return
        
        if checked:
            # Start live analysis from the beginning
            self.mid_point_idx = 0
            self.is_live_analysis_running = True
            dialog.live_analysis_btn.setText("■ Stop Live Analysis")
            
            # Start audio playback if enabled
            if (hasattr(dialog, 'audio_playback_checkbox') and 
                dialog.audio_playback_checkbox.isChecked()):
                # Test if audio is available before trying to play
                try:
                    # This will raise an exception if no audio device is available
                    sd.check_output_settings()
                    self.start_audio_playback(dialog)
                except sd.PortAudioError as e:
                    print(f"No audio device available: {e}")
                    dialog.audio_playback_checkbox.setChecked(False)
            
            # Create timer if it doesn't exist
            if not hasattr(dialog, 'live_analysis_timer'):
                dialog.live_analysis_timer = QTimer()
                dialog.live_analysis_timer.timeout.connect(lambda: self.update_live_analysis(dialog))
            
            # Set initial interval from slider or default
            if hasattr(dialog, 'speed_slider'):
                interval = dialog.speed_slider.value()
            else:
                interval = 100  # Default 100ms
            
            dialog.live_analysis_timer.start(interval)
        else:
            # Stop live analysis
            self.is_live_analysis_running = False
            dialog.live_analysis_btn.setText("▶ Start Live Analysis")
            if hasattr(dialog, 'live_analysis_timer'):
                dialog.live_analysis_timer.stop()
            
            # Stop audio playback if it's active
            try:
                sd.stop()
            except Exception as e:
                print(f"Error stopping audio: {e}")

    def start_audio_playback(self, dialog):
        """Start playing the entire audio from the beginning"""
        try:
            # Print debug information
            print(f"Audio info: shape={self.audio.shape}, dtype={self.audio.dtype}, max={np.max(self.audio)}, min={np.min(self.audio)}")
            print(f"Sample rate: {self.fs}")
            
            # Check if audio data needs normalization
            audio_data = self.audio.astype(np.float32)
            max_val = np.max(np.abs(audio_data))
            print(f"Max absolute value: {max_val}")
            
            if max_val > 1.0:
                print("Normalizing audio data")
                audio_data = audio_data / max_val
            
            # Check number of channels
            channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
            print(f"Channels: {channels}")
            
            # Get default output device
            default_device = sd.default.device
            print(f"Default audio device: {default_device}")
            print(f"Available devices: {sd.query_devices()}")
            
            # Store the audio data in the dialog to prevent garbage collection
            dialog.audio_data = audio_data
            
            # Create a simple playback without callback
            # This is a simpler approach that should work more reliably
            dialog.audio_start_time = time.time()
            
            # Play the audio using a simple non-blocking approach
            sd.play(audio_data, samplerate=self.fs, blocking=False, blocksize=32768, latency='high')            
            print("Audio playback started successfully")
            
        except Exception as e:
            print(f"Error starting audio playback: {e}")
            import traceback
            traceback.print_exc()
                
    def create_stft_plot_dialog(self, figure, waveform_ax, audio_signal):
        """Create a specialized dialog for STFT plots with live analysis button only"""
        try:
            start, end = 0, self.duration
            plot_title = f"STFT&Spectro_{self.base_name}_{self.format_timestamp(start)}-{self.format_timestamp(end)}"

            plot_dialog = QDialog(self)
            plot_dialog.setWindowTitle(plot_title)
            plot_dialog.setAttribute(Qt.WA_DeleteOnClose)
            plot_dialog.plot_id = id(plot_dialog)
            
            # Initialize all cursor-related attributes
            plot_dialog.active_stream = None
            plot_dialog.cursor_timer = QTimer()  # Initialize timer here
            plot_dialog.cursor_timer.setSingleShot(False)
            plot_dialog.cursor_line = None
            plot_dialog.spectrogram_cursor_line = None
            plot_dialog._background_waveform = None
            plot_dialog._background_spectrogram = None

            # Create main layout
            main_layout = QVBoxLayout()

            # ▶ Live analysis button
            live_btn = QPushButton("▶ Start Live Analysis")
            live_btn.setCheckable(True)
            live_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ccffcc;
                    border: 1px solid #ccc;
                    padding: 5px;
                    margin: 5px;
                }
                QPushButton:checked {
                    background-color: #ffcccc;
                }
            """)
            live_btn.clicked.connect(lambda checked: self.toggle_live_analysis(checked, plot_dialog))        
            main_layout.addWidget(live_btn)

            # Speed control layout
            speed_layout = QHBoxLayout()
            speed_label = QLabel("Speed:")
            speed_layout.addWidget(speed_label)
            
            # Add speed control slider
            speed_slider = QSlider(Qt.Horizontal)
            speed_slider.setRange(10, 500)  # 10ms to 500ms
            speed_slider.setValue(100)      # Default 100ms
            speed_slider.setTickPosition(QSlider.TicksBelow)
            speed_slider.setTickInterval(50)
            speed_slider.valueChanged.connect(lambda value: self.update_live_speed(value, plot_dialog))        
            speed_layout.addWidget(speed_slider)
            
            speed_value_label = QLabel("100 ms")
            speed_layout.addWidget(speed_value_label)
            main_layout.addLayout(speed_layout)

            # Audio playback checkbox
            audio_playback_checkbox = QCheckBox("Play audio during analysis")
            audio_playback_checkbox.setChecked(False)
            main_layout.addWidget(audio_playback_checkbox)
            plot_dialog.audio_playback_checkbox = audio_playback_checkbox

            # Connect the checkbox to enable/disable the speed slider
            def toggle_speed_slider(checked):
                speed_slider.setEnabled(not checked)
                speed_label.setEnabled(not checked)
                speed_value_label.setEnabled(not checked)

            audio_playback_checkbox.stateChanged.connect(toggle_speed_slider)

            # Set initial state
            toggle_speed_slider(audio_playback_checkbox.isChecked())

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
            plot_dialog.canvas = canvas  # Store canvas reference for cursor updates
            plot_dialog.speed_slider = speed_slider
            plot_dialog.speed_value_label = speed_value_label

            # Setup span selector - pass the dialog object itself
            self.create_span_selector(waveform_ax, audio_signal, plot_dialog)

            # Define the handle_close function inside this method
            def handle_close():
                # Stop audio playback if active
                try:
                    sd.stop()  # This will stop all audio playback
                except Exception as e:
                    print(f"Error stopping audio: {e}")
                
                # Stop and disconnect cursor timer
                if hasattr(plot_dialog, 'live_analysis_timer') and plot_dialog.live_analysis_timer is not None:
                    try:
                        plot_dialog.live_analysis_timer.stop()
                        # Disconnect all connections from the timer
                        try:
                            plot_dialog.live_analysis_timer.timeout.disconnect()
                        except TypeError:
                            # This might fail if no connections exist, which is fine
                            pass
                    except Exception as e:
                        print(f"Error stopping timer: {e}")
                
                # Clean up other resources
                self.cleanup_plot_window(plot_dialog.plot_id)

            plot_dialog.finished.connect(handle_close)

            # Track open windows
            self.plot_windows.append(plot_dialog)

            # Debug print
            print(f"Total plot windows: {len(self.plot_windows)}")
            for i, window in enumerate(self.plot_windows):
                print(f"Window {i+1} title: {window.windowTitle()}")

            plot_dialog.show()
            return plot_dialog
            
        except Exception as e:
            print(f"Error creating STFT plot dialog: {e}")
            import traceback
            traceback.print_exc()
            return None
       
    def update_live_speed(self, value, dialog=None):
        """Update the live analysis speed based on slider value"""
        if dialog is None and hasattr(self, 'current_figure'):
            for window in self.plot_windows:
                if hasattr(window, 'figure') and window.figure == self.current_figure:
                    dialog = window
                    break
        
        if dialog is None:
            return
        
        dialog.live_analysis_interval = value
        
        # Only update the timer interval if audio playback is not enabled
        if (hasattr(dialog, 'audio_playback_checkbox') and 
            not dialog.audio_playback_checkbox.isChecked() and
            hasattr(dialog, 'live_analysis_timer') and 
            dialog.live_analysis_timer.isActive()):
            dialog.live_analysis_timer.setInterval(value)
        
        # Update the speed label
        if hasattr(dialog, 'speed_value_label'):
            dialog.speed_value_label.setText(f"{value} ms")

    def plot_stft_spect(self):
        """STFT + Spectrogram with interactive window selection and live analysis button"""
        try:
            # Get current font size (default to 12 if not set)
            fontsize = getattr(self, 'current_font_size', 12)
            
            # Set style before creating figure
            plt.style.use('default')
            plt.rcParams.update({'font.size': fontsize})

            wind_size = float(self.window_size.text())
            overlap = float(self.overlap.text())
            nfft = int(self.nfft.currentText())
            min_freq = int(self.min_freq.text())
            max_freq = int(self.max_freq.text())
            self.min_freq_val = min_freq
            self.max_freq_val = max_freq

            draw_style = self.draw_style.currentIndex() + 1
            
            # STFT analysis properties - set these first
            self.wind_size_samples = int(wind_size * self.fs)
            self.hop_size = self.wind_size_samples - int(overlap * self.fs)
            self.window = self.get_window(self.wind_size_samples)
            self.nfft_val = nfft

            # Calculate the global range of STFT values for consistent y-axis scaling
            # We'll analyze multiple representative windows to find the true range
            self.global_stft_min = float('inf')
            self.global_stft_max = float('-inf')

            # Sample windows at regular intervals throughout the audio
            sample_indices = np.linspace(0, len(self.audio) - self.wind_size_samples, 
                                        min(100, len(self.audio) // self.wind_size_samples), 
                                        dtype=int)
            
            for i in sample_indices:
                start = i
                end = i + self.wind_size_samples
                
                # Get the audio segment
                audio_segment = self.audio[start:end]
                window_segment = self.window[:len(audio_segment)]
                
                # Ensure perfect alignment
                if len(audio_segment) < len(window_segment):
                    window_segment = window_segment[:len(audio_segment)]
                elif len(window_segment) < len(audio_segment):
                    audio_segment = audio_segment[:len(window_segment)]
                
                windowed = audio_segment * window_segment
                
                # Compute STFT with padding if needed
                if len(windowed) < self.nfft_val:
                    windowed = np.pad(windowed, (0, self.nfft_val - len(windowed)))
                stft = np.abs(np.fft.fft(windowed, self.nfft_val)[:self.nfft_val//2])
                
                # Convert to dB
                stft_db = 20 * np.log10(stft + 1e-10)
                
                # Update global min and max
                self.global_stft_min = min(self.global_stft_min, np.min(stft_db))
                self.global_stft_max = max(self.global_stft_max, np.max(stft_db))
            
            
            # Add a very generous margin to ensure no clipping
            # Use a combination of percentage and fixed margin
            range_size = self.global_stft_max - self.global_stft_min
            
            '''
            # If the range is very small, use a fixed margin
            if range_size < 10:
                margin = 5  # 5 dB fixed margin
            else:
                margin = range_size * 0.5  # 50% margin'''

            margin = range_size * 1.0


            # Add a safety margin to ensure no clipping
            margin = (self.global_stft_max - self.global_stft_min) * 0.2  # 20% margin
            self.global_stft_min -= margin
            self.global_stft_max += margin
            
            # Ensure we have a reasonable range even for very quiet signals
            if self.global_stft_max - self.global_stft_min < 40:  # Less than 20 dB range
                self.global_stft_min = -120
                self.global_stft_max = 0


            
            # Now create the figure
            self.current_figure = plt.figure(figsize=(12, 8))
            gs = plt.GridSpec(3, 2, width_ratios=[15, 1], height_ratios=[1, 1, 1.5], hspace=0.4)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[1, 0])
            ax3 = plt.subplot(gs[2, 0], sharex=ax1)
            cbar_ax = plt.subplot(gs[:, 1])

            # Custom coordinate display
            def format_time_amp(x, y):
                return f"time = {x:.2f} s, amplitude = {y:.3f}"

            def format_freq_db(x, y):
                return f"freq = {x:.1f} Hz, magnitude = {y:.1f} dB"

            def format_time_freq(x, y):
                return f"time = {x:.2f} s, freq = {y:.1f} Hz"

            ax1.format_coord = format_time_amp   # time-domain waveform
            ax2.format_coord = format_freq_db    # FFT window
            ax3.format_coord = format_time_freq  # spectrogram

            
            self.current_figure.suptitle('STFT + Spectrogram', y=0.98)

            self.is_live_analysis_running = False
            
            # Ensure time and audio arrays match
            if len(self.time) > len(self.audio):
                self.time = self.time[:len(self.audio)]
            elif len(self.audio) > len(self.time):
                self.audio = self.audio[:len(self.time)]
            
            # CHECK FOR START
            self.mid_point_idx = len(self.audio) // 2  # Start in middle
            
            # Create initial spectrogram image
            if draw_style == 1:
                D = librosa.stft(self.audio, n_fft=self.nfft_val, hop_length=self.hop_size,
                                win_length=self.wind_size_samples, window=self.window)
                self.S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                self.img = librosa.display.specshow(self.S_db, x_axis='time', y_axis='linear',
                                                sr=self.fs, hop_length=self.hop_size,
                                                ax=ax3)
                # Calculate global range from the complete STFT
                self.global_stft_min = np.min(self.S_db)
                self.global_stft_max = np.max(self.S_db)
                
                # Manually set axis [y]frequency x[audio_duration] range
                ax3.set_ylim([min_freq, max_freq])
                ax3.set_xlim([0, len(self.audio) / self.fs])
            else:
                S = librosa.feature.melspectrogram(y=self.audio, sr=self.fs,
                                                n_fft=self.nfft_val, hop_length=self.hop_size,
                                                win_length=self.wind_size_samples,
                                                window=self.window, fmin=min_freq,
                                                fmax=max_freq)
                self.S_db = librosa.power_to_db(S, ref=np.max)
                self.img = librosa.display.specshow(self.S_db, x_axis='time', y_axis='mel',
                                                sr=self.fs, hop_length=self.hop_size,
                                                ax=ax3)
                # Calculate global range from the complete STFT
                self.global_stft_min = np.min(self.S_db)
                self.global_stft_max = np.max(self.S_db)
                
                # Manually set axis [y]frequency x[audio_duration] range
                ax3.set_ylim([min_freq, max_freq])
                ax3.set_xlim([0, len(self.audio) / self.fs])

            # Add a margin to the global range
            margin = (self.global_stft_max - self.global_stft_min) * 0.1
            self.global_stft_min -= margin
            self.global_stft_max += margin

            # Make spectrogram labels more readable
            ax3.set_ylabel('Frequency (Hz)', fontsize=10)
            ax3.tick_params(axis='both', which='major', labelsize=8)
            
            duration = len(self.audio) / self.fs
            ax1.set_xlim([0, duration])
            ax3.set_xlim([0, duration])

            # Create colorbar
            self.cbar = self.current_figure.colorbar(self.img, cax=cbar_ax, format="%+2.0f dB")
            
            # Initial plot
            self.update_stft_spect_plot(ax1, ax2, ax3)
            
            self.current_figure.canvas.mpl_connect(
                'button_press_event', 
                lambda e: self.on_window_click_spect(e, ax1, ax2, ax3)
            )
            
            # Create and show the plot dialog with live analysis button
            plot_dialog = self.create_stft_plot_dialog(self.current_figure, ax1, self.audio)

            # Store spectrogram axis reference only if dialog was created successfully
            if plot_dialog is not None:
                plot_dialog.spectrogram_ax = ax3
            else:
                print("Warning: Failed to create plot dialog")
                return None  # Return early if dialog creation failed

            return plot_dialog

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
        """Update plot with proper array handling and fixed y-axis"""
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
        
        # Convert to dB
        stft_db = 20 * np.log10(stft + 1e-10)
        
        # Plot time domain with highlighted window
        ax1.plot(self.time, self.audio)
        ax1.set_xlim([0, len(self.audio) / self.fs])

        ax1.axvspan(time_segment[0], time_segment[-1], color='lightblue', alpha=0.3)
        if self.mid_point_idx < len(self.time):
            ax1.axvline(self.time[self.mid_point_idx], color='red', ls='--')
                
        # Plot STFT with fixed y-axis using the global range
        ax2.plot(freqs, stft_db)
        

        current_max = np.max(stft_db)
        ylim_top = max(self.global_stft_max, current_max) + 3  # add margin
        ax2.set(
            xlim=[self.min_freq_val, self.max_freq_val], 
            ylim=[self.global_stft_min, ylim_top],
            xlabel='Frequency (Hz)', 
            ylabel='Magnitude (dB)'
        )


        # Update spectrogram data while keeping fixed dB range
        self.img.set_array(self.S_db)
        self.img.set_clim(self.global_stft_min, self.global_stft_max)
                
        self.current_figure.canvas.draw()

    # Short Time Energy
    
    def plot_ste(self):
        # Get current font size (default to 12 if not set)
        fontsize = getattr(self, 'current_font_size', 12)
        
        self.update_beta_state()

        # Set style before creating figure
        plt.style.use('default')
        plt.rcParams.update({'font.size': fontsize})

        show_pitch = self.show_pitch.isChecked()
        wind_size = float(self.window_size.text())
        overlap = float(self.overlap.text())
        window_type = self.window_type.currentText()
        beta = float(self.beta.text()) if window_type == 'Kaiser' else 0

        # Get parameters from UI
        min_pitch = float(self.min_pitch.text())
        max_pitch = float(self.max_pitch.text())
        
        self.current_figure, ax = plt.subplots(2, figsize=(12,6), sharex=True)
        title = 'Short-Time Energy with Pitch' if show_pitch else 'Short-Time Energy'
        self.current_figure.suptitle(title)
        
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

        if window_type == 'Kaiser':
            try:
                beta = float(self.beta.text())
                if beta < 0 or beta > 50:
                    beta = 14.0
                    self.beta.setText("14.0")
            except ValueError:
                beta = 14.0
                self.beta.setText("14.0")
        else:
            beta = 0
        
        # Calculate STE with proper hop size and dB conversion
        ste = []
        time_points = []
        for i in range(0, len(self.audio) - wind_size_samples, hop_size):
            segment = self.audio[i:i+wind_size_samples] * window
            energy = 10 * np.log10(np.mean(segment**2) + 1e-12)
            ste.append(energy)
            time_points.append(self.time[i + wind_size_samples//2])
        
        # Plot original waveform
        ax[0].plot(self.time, self.audio)
        ax[0].set(ylabel='Amplitude')
        
        # Plot STE in dB
        ste_line = ax[1].plot(time_points, ste, color='blue', linewidth=1.5, label='STE')[0]
        ax[1].set(xlim=[0, self.duration], xlabel='Time (s)', ylabel='Short-Time Energy (dB)')


        # Add pitch tracking if enabled
        if show_pitch:
            try:
                # Calculate pitch
                pitch, pitch_values = self.calculate_pitch()
                pitch_times = librosa.times_like(pitch_values, sr=self.fs, hop_length=hop_size)
                
                # Debug: Print pitch values to see what we're getting
                print(f"Pitch values range: {np.min(pitch_values):.1f} to {np.max(pitch_values):.1f} Hz")
                print(f"Pitch times range: {np.min(pitch_times):.2f} to {np.max(pitch_times):.2f} s")
                print(f"Number of pitch points: {len(pitch_values)}")
                
                # Create twin axis for pitch
                ax_pitch = ax[1].twinx()
                pitch_line = ax_pitch.plot(pitch_times, pitch_values, color='red', linewidth=1.5, 
                                         alpha=0.9, label='Pitch')[0]
                ax_pitch.set_ylabel('Pitch (Hz)', color='red')
                ax_pitch.tick_params(axis='y', labelcolor='red')
                
                # Set pitch axis limits based on actual data (with padding)
                pitch_nonzero = pitch_values[pitch_values > 0]  # Filter out zeros/unvoiced
                if len(pitch_nonzero) > 0:
                    pitch_min = max(min_pitch, np.min(pitch_nonzero) * 0.8)  # Minimum 50 Hz
                    pitch_max = min(max_pitch, np.max(pitch_nonzero) * 1.2)  # Maximum 500 Hz
                    ax_pitch.set_ylim([pitch_min, pitch_max])
                else:
                    ax_pitch.set_ylim([50, 300])  # Default range if no pitch detected
                
                # Add legend
                lines = [ste_line, pitch_line]
                labels = [l.get_label() for l in lines]
                ax[1].legend(lines, labels, loc='upper right')
                
                # Enhanced coordinate display
                def format_time_energy_pitch(x, y):
                    idx = np.argmin(np.abs(np.array(pitch_times) - x))
                    pitch_val = pitch_values[idx] if idx < len(pitch_values) else 0
                    return f"time = {x:.2f} s, energy = {y:.3f} dB, pitch = {pitch_val:.1f} Hz"
                ax[1].format_coord = format_time_energy_pitch
                
            except Exception as e:
                print(f"Pitch plotting error: {e}")
                # Fallback to regular STE display
                def format_time_energy(x, y):
                    return f"time = {x:.2f} s, energy = {y:.3f} dB"
                ax[1].format_coord = format_time_energy
        else:
            def format_time_energy(x, y):
                return f"time = {x:.2f} s, energy = {y:.3f} dB"
            ax[1].format_coord = format_time_energy

        def format_time_amp(x, y):
            return f"time = {x:.2f} s, amplitude = {y:.3f}"
        ax[0].format_coord = format_time_amp

        self.show_plot_window(self.current_figure, ax[0], self.audio)


    # Spectral Centroid

    def plot_spectral_centroid(self):

        # Get current font size (default to 12 if not set)
        fontsize = getattr(self, 'current_font_size', 12)
        
        # Set style before creating figure
        plt.style.use('default')
        plt.rcParams.update({'font.size': fontsize})

        wind_size = float(self.window_size.text())
        overlap = float(self.overlap.text())
        nfft = int(self.nfft.currentText())
        min_freq = int(self.min_freq.text())
        max_freq = int(self.max_freq.text())
        draw_style = self.draw_style.currentIndex() + 1

        self.current_figure = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(3, 2, width_ratios=[20, 1])

        ax1 = self.current_figure.add_subplot(gs[0, 0])  # Waveform
        ax2 = self.current_figure.add_subplot(gs[1, 0])  # PSD
        ax3 = self.current_figure.add_subplot(gs[2, 0])  # Spectrogram
        cax = self.current_figure.add_subplot(gs[2, 1])  # Colorbar

        self.current_figure.suptitle('Spectral Centroid')

        # Store analysis parameters as attributes
        self.sc_wind_size_samples = int(wind_size * self.fs)
        self.sc_hop_size = self.sc_wind_size_samples - int(overlap * self.fs)
        self.sc_window = self.get_window(self.sc_wind_size_samples)
        self.sc_mid_point_idx = len(self.audio) // 2  # Start in middle

        # Initial plot
        self.update_spectral_centroid_plot(ax1, ax2, ax3, cax, draw_style, min_freq, max_freq, nfft)

        # Connect mouse click event
        self.current_figure.canvas.mpl_connect(
            'button_press_event',
            lambda e: self.on_sc_window_click(e, ax1, ax2, ax3, cax, draw_style, min_freq, max_freq, nfft)
        )

        self.show_plot_window(self.current_figure, ax1, self.audio)

    def calculate_sc(self, segment):
        magnitudes = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), 1/self.fs)
        magnitudes = magnitudes ** 2
        return np.sum(magnitudes * freqs) / np.sum(magnitudes)

    def on_sc_window_click(self, event, ax1, ax2, ax3, cax, draw_style, min_freq, max_freq, nfft):
        """Handle ONLY simple clicks for spectral centroid window movement"""
        if event.inaxes != ax1 or event.button != 1:
            return
            
        # Skip if this is part of a drag operation
        if hasattr(event, 'pressed') and event.pressed:
            return
            
        # Skip if we're currently dragging a span selector
        if hasattr(self, '_is_dragging_span') and self._is_dragging_span:
            return
            
        # Move analysis window to click position
        self.sc_mid_point_idx = np.searchsorted(self.time, event.xdata)
        self.sc_mid_point_idx = min(self.sc_mid_point_idx, len(self.time) - 1)
        
        # Redraw with new position
        self.update_spectral_centroid_plot(ax1, ax2, ax3, cax, draw_style, min_freq, max_freq, nfft)
    
    def update_spectral_centroid_plot(self, ax1, ax2, ax3, cax, draw_style, min_freq, max_freq, nfft):
        """Update all plots with current window position"""
        # Clear previous plots

        for ax in [ax1, ax2, ax3, cax]:
            ax.clear()
        
        # Get current window segment
        start = max(0, self.sc_mid_point_idx - self.sc_wind_size_samples//2)
        end = min(len(self.audio), self.sc_mid_point_idx + self.sc_wind_size_samples//2)
        audio_segment = self.audio[start:end]
        window_segment = self.sc_window[:len(audio_segment)]
        windowed_segment = audio_segment * window_segment

        # Calculate spectral centroid for this segment
        spectral_centroid = self.calculate_sc(windowed_segment)
        sc_value = f"{spectral_centroid:.2f}"

        # === Waveform ===
        ax1.plot(self.time, self.audio)
        ax1.axvspan(self.time[start], self.time[end-1], color='silver', alpha=0.5)
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(self.time[0], self.time[-1])

        _, freqs = ax2.psd(windowed_segment, NFFT=self.sc_wind_size_samples, Fs=self.fs,
                           window=self.sc_window, noverlap=0)

        ax2.axvline(x=spectral_centroid, color='r')
        ax2.set_xlim([0, max(freqs)])
        ax2.set_ylabel("Power")
        ax2.set_title(f"Spectral Centroid: {sc_value} Hz")

        # === Spectrogram ===
        if draw_style == 1:
            D = librosa.stft(self.audio, n_fft=nfft, hop_length=self.sc_hop_size,
                             win_length=self.sc_wind_size_samples, window=self.sc_window)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear',
                                           sr=self.fs, hop_length=self.sc_hop_size,
                                           fmin=min_freq, fmax=max_freq, ax=ax3)
            ax3.set_ylim([min_freq, max_freq])
        else:
            S = librosa.feature.melspectrogram(y=self.audio, sr=self.fs,
                                               n_fft=nfft, hop_length=self.sc_hop_size,
                                               win_length=self.sc_wind_size_samples,
                                               window=self.sc_window, fmin=min_freq,
                                               fmax=max_freq)
            S_db = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                           sr=self.fs, hop_length=self.sc_hop_size,
                                           fmin=min_freq, fmax=max_freq, ax=ax3)

        # Overlay spectral centroid
        sc = librosa.feature.spectral_centroid(y=self.audio, sr=self.fs,
                                               n_fft=nfft, hop_length=self.sc_hop_size,
                                               win_length=self.sc_wind_size_samples)
        times = librosa.times_like(sc, sr=self.fs, hop_length=self.sc_hop_size)
        ax3.plot(times, sc[0], color='w', linewidth=1.5)
        ax3.set_ylabel("Freq (Hz)")
        ax3.set_xlabel("Time (s)")

        # Custom coordinate display
        def format_time_amp(x, y):
            return f"time = {x:.2f} s, amplitude = {y:.3f}"

        def format_freq_db(x, y):
            return f"freq = {x:.1f} Hz, magnitude = {y:.1f} dB"

        def format_time_freq(x, y):
            return f"time = {x:.2f} s, freq = {y:.1f} Hz"

        ax1.format_coord = format_time_amp   # time-domain waveform
        ax2.format_coord = format_freq_db    # FFT window
        ax3.format_coord = format_time_freq  # spectrograms

        # Colorbar
        self.current_figure.colorbar(img, cax=cax, format="%+2.0f dB")
        plt.tight_layout(rect=[0, 0, 0.97, 0.95])
        
        self.current_figure.canvas.draw()

    # Filtered section.

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

    def plot_filtered_waveform(self, filter_type, filtered_signal):
        """Plot original and filtered waveforms with proper span selectors in same window"""
        # Get current font size (default to 12 if not set)
        fontsize = getattr(self, 'current_font_size', 12)
        
        # Set style before creating figure
        plt.style.use('default')
        plt.rcParams.update({'font.size': fontsize})


        self.current_figure, ax = plt.subplots(2, figsize=(12, 6))
        self.current_figure.suptitle(f'Filtered Signal ({filter_type}) - Waveform')

        # Plot original
        ax[0].plot(self.time, self.audio)
        ax[0].set(xlim=[0, self.duration], title='Original Signal')

        # Plot filtered
        ax[1].plot(self.time, filtered_signal)
        ax[1].set(xlim=[0, self.duration], title='Filtered Signal')

        # One shared dialog window with both axes
        shared_dialog = self.show_plot_window(self.current_figure, create_selector=False)

        # Assign custom attributes to distinguish axes
        shared_dialog.original_ax = ax[0]
        shared_dialog.filtered_ax = ax[1]

        # Custom coordinate display
        def format_time_amp(x, y):
            return f"time = {x:.2f} s, amplitude = {y:.3f}"
        ax[0].format_coord = format_time_amp   # time-domain waveform
        ax[1].format_coord = format_time_amp   # time-domain waveform
        

            # Create two separate span selectors, store them under the same dialog.plot_id
        self.create_span_selector(ax[0], self.audio, shared_dialog, tag='original')
        self.create_span_selector(ax[1], filtered_signal, shared_dialog, tag='filtered')

    def plot_filtered_spectrogram(self, filter_type, filtered_signal):
        """Plot original and filtered spectrograms in shared window with independent span selectors."""
        try:
            # Get current font size (default to 12 if not set)
            fontsize = getattr(self, 'current_font_size', 12)
            
            # Set style before creating figure
            plt.style.use('default')
            plt.rcParams.update({'font.size': fontsize})

            # --- PARAMETERS ---
            params = self.validate_spectrogram_parameters()
            wind_size_samples = params['wind_size_samples']
            hop_size = params['hop_size']
            nfft = params['nfft']
            min_freq = params['min_freq']
            max_freq = params['max_freq']

            show_pitch = self.show_pitch.isChecked()
            draw_style = self.draw_style.currentIndex() + 1
            window = self.get_window(wind_size_samples)

            # Create figure and layout
            self.current_figure = plt.figure(figsize=(12, 8))
            gs = plt.GridSpec(2, 1, height_ratios=[1, 1])

            def compute_and_plot(ax, signal, title, pitch_color):
                """Compute and plot the spectrogram (linear or mel) with optional pitch."""
                if draw_style == 1:  # Linear
                    D = librosa.stft(signal, n_fft=nfft, hop_length=hop_size,
                                     win_length=wind_size_samples, window=window)
                    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear',
                                                   sr=self.fs, hop_length=hop_size,
                                                   fmin=min_freq, fmax=max_freq, ax=ax)
                    ax.set_ylim([min_freq, max_freq])
                else:  # Mel
                    S = librosa.feature.melspectrogram(y=signal, sr=self.fs,
                                                       n_fft=nfft, hop_length=hop_size,
                                                       win_length=wind_size_samples,
                                                       window=window, fmin=min_freq,
                                                       fmax=max_freq)
                    S_db = librosa.power_to_db(S, ref=np.max)
                    img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                                   sr=self.fs, hop_length=hop_size,
                                                   fmin=min_freq, fmax=max_freq, ax=ax)

                duration = librosa.frames_to_time(S_db.shape[1], sr=self.fs, hop_length=hop_size)
                ax.set_xlim([0, duration])
                ax.set(title=title)

                if show_pitch:
                    _, pitch_smoothed = self.calculate_pitch(signal=signal)
                    pitch_times = librosa.times_like(pitch_smoothed, sr=self.fs, hop_length=hop_size)
                    ax.plot(pitch_times, pitch_smoothed, '-', color=pitch_color,
                            linewidth=2, alpha=0.9, label='Pitch')

                return img, S_db

            # Plot original spectrogram
            ax0 = plt.subplot(gs[0])
            img0, _ = compute_and_plot(ax0, self.audio, 'Original Signal Spectrogram', pitch_color='lime')

            # Plot filtered spectrogram
            ax1 = plt.subplot(gs[1])
            img1, _ = compute_and_plot(ax1, filtered_signal,
                                       f'Filtered Signal Spectrogram ({filter_type})', pitch_color='cyan')

            self.current_figure.tight_layout()

            # Shared plot dialog for both axes
            shared_dialog = self.show_plot_window(self.current_figure, create_selector=False)

            # Tag each axis to distinguish selectors
            shared_dialog.original_ax = ax0
            shared_dialog.filtered_ax = ax1

            def format_time_freq(x, y):
                return f"time = {x:.2f} s, freq = {y:.1f} Hz"
            ax0.format_coord = format_time_freq  # spectrogram
            ax1.format_coord = format_time_freq  # spectrogram

            # Independent span selectors on both plots with tags
            self.create_span_selector(ax0, self.audio, shared_dialog, tag='original')
            self.create_span_selector(ax1, filtered_signal, shared_dialog, tag='filtered')

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


    # Helper plot methods

    def create_span_selector(self, ax, audio_signal, plot_dialog, tag='default'):
        def onselect(xmin, xmax):
            if xmin == xmax:
                return

            # Store original limits
            original_ylim = ax.get_ylim()
            if hasattr(plot_dialog, 'spectrogram_ax') and plot_dialog.spectrogram_ax is not None:
                original_spec_ylim = plot_dialog.spectrogram_ax.get_ylim()

            # Cleanup previous session
            if hasattr(plot_dialog, 'active_stream') and plot_dialog.active_stream is not None:
                try:
                    plot_dialog.active_stream.stop()
                    plot_dialog.active_stream.close()
                except Exception:
                    pass
                plot_dialog.active_stream = None

            # Timer cleanup with safe disconnection
            if hasattr(plot_dialog, 'cursor_timer') and plot_dialog.cursor_timer is not None:
                try:
                    plot_dialog.cursor_timer.stop()
                    # Safe disconnection by checking if connected first
                    receivers = plot_dialog.cursor_timer.receivers(plot_dialog.cursor_timer.timeout)
                    if receivers > 0:
                        plot_dialog.cursor_timer.timeout.disconnect()
                except Exception as e:
                    print(f"Timer cleanup error: {e}")

            # Remove previous visual elements
            for element in ['cursor_line', 'spectrogram_cursor_line', 'visible_span_patch']:
                if hasattr(plot_dialog, element) and getattr(plot_dialog, element) is not None:
                    try:
                        getattr(plot_dialog, element).remove()
                    except Exception:
                        pass
                    setattr(plot_dialog, element, None)

            # Create new selection patch
            plot_dialog.visible_span_patch = patches.Rectangle(
                (xmin, original_ylim[0]),
                xmax - xmin,
                original_ylim[1] - original_ylim[0],
                linewidth=0,
                edgecolor=None,
                facecolor='red',
                alpha=0.3,
                zorder=1
            )
            ax.add_patch(plot_dialog.visible_span_patch)
            ax.set_ylim(original_ylim)  # Maintain original scale

            # Create new cursors
            plot_dialog.cursor_line, = ax.plot([xmin, xmin], original_ylim, 'r', linewidth=1.2)
            if hasattr(plot_dialog, 'spectrogram_ax') and plot_dialog.spectrogram_ax is not None:
                plot_dialog.spectrogram_cursor_line, = plot_dialog.spectrogram_ax.plot(
                    [xmin, xmin], original_spec_ylim, 'r', linewidth=1.2)

            # Audio setup
            start_sample = int(xmin * self.fs)
            end_sample = int(xmax * self.fs)
            segment = audio_signal[start_sample:end_sample].astype(np.float32)
            total_len = len(segment)
            stream_pos = 0

            # Save backgrounds after drawing
            canvas = ax.figure.canvas
            canvas.draw()
            plot_dialog._background_waveform = canvas.copy_from_bbox(ax.bbox)
            if hasattr(plot_dialog, 'spectrogram_ax') and plot_dialog.spectrogram_ax is not None:
                try:
                    plot_dialog._background_spectrogram = canvas.copy_from_bbox(plot_dialog.spectrogram_ax.bbox)
                except Exception as e:
                    print(f"Background save error: {e}")
                    plot_dialog._background_spectrogram = None

            def update_cursor():
                nonlocal stream_pos
                if not plot_dialog.isVisible():
                    if hasattr(plot_dialog, 'cursor_timer'):
                        plot_dialog.cursor_timer.stop()
                    return

                elapsed = time.time() - t0
                current_time = xmin + elapsed

                if current_time >= xmax:
                    plot_dialog.cursor_timer.stop()
                    return

                # Update cursors
                plot_dialog.cursor_line.set_xdata([current_time, current_time])
                if hasattr(plot_dialog, 'spectrogram_cursor_line') and plot_dialog.spectrogram_cursor_line is not None:
                    plot_dialog.spectrogram_cursor_line.set_xdata([current_time, current_time])

                # Redraw
                try:
                    canvas.restore_region(plot_dialog._background_waveform)
                    ax.draw_artist(plot_dialog.visible_span_patch)
                    ax.draw_artist(plot_dialog.cursor_line)
                    canvas.blit(ax.bbox)
                    
                    if (hasattr(plot_dialog, '_background_spectrogram') and 
                       (plot_dialog._background_spectrogram is not None)):
                        canvas.restore_region(plot_dialog._background_spectrogram)
                        plot_dialog.spectrogram_ax.draw_artist(plot_dialog.spectrogram_cursor_line)
                        canvas.blit(plot_dialog.spectrogram_ax.bbox)
                except Exception as e:
                    print(f"Drawing error: {e}")

            def callback(outdata, frames, time_info, status):
                nonlocal stream_pos
                if status:
                    print(status)

                remaining = total_len - stream_pos
                if remaining <= 0:
                    raise sd.CallbackStop()

                n = min(frames, remaining)
                outdata[:n] = segment[stream_pos:stream_pos + n].reshape(-1, 1)
                if n < frames:
                    outdata[n:] = 0
                stream_pos += n

            # Start playback
            try:
                plot_dialog.active_stream = sd.OutputStream(
                    samplerate=self.fs,
                    channels=1,
                    dtype='float32',
                    callback=callback,
                    blocksize=1024
                )
                plot_dialog.active_stream.start()
                
                t0 = time.time()
                if not hasattr(plot_dialog, 'cursor_timer') or plot_dialog.cursor_timer is None:
                    plot_dialog.cursor_timer = QTimer()
                plot_dialog.cursor_timer.timeout.connect(update_cursor)
                plot_dialog.cursor_timer.start(30)
                
            except Exception as e:
                print(f"Playback error: {e}")

        span_selector = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                                        props=dict(alpha=0.3, facecolor='red'))

        if not hasattr(self, 'span_selectors'):
            self.span_selectors = {}
        if plot_dialog.plot_id not in self.span_selectors:
            self.span_selectors[plot_dialog.plot_id] = {}

        # Store selector using the tag (e.g., 'original', 'filtered')
        self.span_selectors[plot_dialog.plot_id][tag] = span_selector

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

        # Initialize all cursor-related attributes
        plot_dialog.active_stream = None
        plot_dialog.cursor_timer = QTimer()  # Initialize timer
        plot_dialog.cursor_timer.setSingleShot(False)
        plot_dialog.cursor_line = None
        plot_dialog.spectrogram_cursor_line = None
        plot_dialog._background_waveform = None
        plot_dialog._background_spectrogram = None
        plot_dialog.visible_span_patch = None

        # Store references
        plot_dialog.figure = figure
        plot_dialog.waveform_ax = waveform_ax
        self.current_figure = figure

        layout = QVBoxLayout()
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, plot_dialog)


        # Add playback controls
        control_layout = QHBoxLayout()
        
        # Start time spinbox
        start_label = QLabel("Start (s):")
        plot_dialog.start_spin = QDoubleSpinBox()
        plot_dialog.start_spin.setRange(0, self.duration)
        plot_dialog.start_spin.setValue(0)
        plot_dialog.start_spin.setSingleStep(0.1)
        plot_dialog.start_spin.setDecimals(3)
        
        # End time spinbox
        end_label = QLabel("End (s):")
        plot_dialog.end_spin = QDoubleSpinBox()
        plot_dialog.end_spin.setRange(0, self.duration)
        plot_dialog.end_spin.setValue(self.duration)
        plot_dialog.end_spin.setSingleStep(0.1)
        plot_dialog.end_spin.setDecimals(3)
        
        # Play button
        play_button = QPushButton("Play Selection")
        play_button.clicked.connect(lambda: self.play_audio_selection(
            plot_dialog, 
            waveform_ax, 
            audio_signal,
            plot_dialog.start_spin.value(),
            plot_dialog.end_spin.value()
        ))
        
        # Add widgets to control layout
        control_layout.addWidget(start_label)
        control_layout.addWidget(plot_dialog.start_spin)
        control_layout.addWidget(end_label)
        control_layout.addWidget(plot_dialog.end_spin)
        control_layout.addWidget(play_button)

        # Add to main layout
        layout.addWidget(toolbar)
        layout.addLayout(control_layout)  # Add controls below toolbar
        layout.addWidget(canvas)
        plot_dialog.setLayout(layout)

        def handle_close():
            # Stream cleanup
            if hasattr(plot_dialog, 'active_stream') and plot_dialog.active_stream is not None:
                try:
                    plot_dialog.active_stream.stop()
                    plot_dialog.active_stream.close()
                except Exception as e:
                    print(f"Error stopping stream: {e}")
                finally:
                    plot_dialog.active_stream = None

            # Timer cleanup - safer disconnection
            if hasattr(plot_dialog, 'cursor_timer') and plot_dialog.cursor_timer is not None:
                try:
                    plot_dialog.cursor_timer.stop()
                    # Only disconnect if connected
                    if plot_dialog.cursor_timer.receivers(plot_dialog.cursor_timer.timeout) > 0:
                        plot_dialog.cursor_timer.timeout.disconnect()
                except Exception as e:
                    print(f"Error stopping timer: {e}")
                finally:
                    plot_dialog.cursor_timer = None

            if hasattr(self.controller, 'update_windows_menu'):
                self.controller.update_windows_menu()

            # Visual elements cleanup
            for cursor_attr in ['cursor_line', 'spectrogram_cursor_line', 'visible_span_patch']:
                if hasattr(plot_dialog, cursor_attr) and getattr(plot_dialog, cursor_attr) is not None:
                    try:
                        getattr(plot_dialog, cursor_attr).remove()
                    except Exception as e:
                        print(f"Error removing {cursor_attr}: {e}")
                    finally:
                        setattr(plot_dialog, cursor_attr, None)

            # Clean up selectors through the main window
            self.on_plot_window_close(plot_dialog.plot_id)
            
            # Remove from windows list if it exists
            if hasattr(self, 'plot_windows') and plot_dialog in self.plot_windows:
                print(f"Removing a window total plot windows: {len(self.plot_windows)}")
                self.plot_windows.remove(plot_dialog)
                # Debug print
                print(f"Removed: Total plot windows: {len(self.plot_windows)}")
                for i, window in enumerate(self.plot_windows):
                    print(f"Window {i+1} title: {window.windowTitle()}")

                    
        plot_dialog.finished.connect(handle_close)

        if create_selector and waveform_ax is not None and audio_signal is not None:
            self.create_span_selector(waveform_ax, audio_signal, plot_dialog)

        self.plot_windows.append(plot_dialog)
        if hasattr(self.controller, 'update_windows_menu'):
            self.controller.update_windows_menu()
        # Debug print
        print(f"Total plot windows: {len(self.plot_windows)}")
        for i, window in enumerate(self.plot_windows):
            print(f"Window {i+1} title: {window.windowTitle()}")

        plot_dialog.show()
        return plot_dialog

    def play_audio_selection(self, plot_dialog, ax, audio_signal, start_time, end_time):
        """Play audio segment while maintaining proper y-axis limits"""
        # Store original y-limits before any modifications
        original_ylim = ax.get_ylim()
        if hasattr(plot_dialog, 'spectrogram_ax') and plot_dialog.spectrogram_ax is not None:
            original_spec_ylim = plot_dialog.spectrogram_ax.get_ylim()
        
        # Cleanup any existing playback
        if hasattr(plot_dialog, 'active_stream') and plot_dialog.active_stream is not None:
            try:
                plot_dialog.active_stream.stop()
                plot_dialog.active_stream.close()
            except Exception as e:
                print(f"Error stopping previous stream: {e}")
            plot_dialog.active_stream = None
        
        if hasattr(plot_dialog, 'cursor_timer') and plot_dialog.cursor_timer is not None:
            plot_dialog.cursor_timer.stop()
            if plot_dialog.cursor_timer.receivers(plot_dialog.cursor_timer.timeout) > 0:
                plot_dialog.cursor_timer.timeout.disconnect()
        
        # Clear previous visual elements
        for element in ['cursor_line', 'spectrogram_cursor_line', 'visible_span_patch']:
            if hasattr(plot_dialog, element) and getattr(plot_dialog, element) is not None:
                try:
                    getattr(plot_dialog, element).remove()
                except Exception:
                    pass
                setattr(plot_dialog, element, None)
        
        # Validate times
        if start_time >= end_time:
            print("Invalid time range")
            return
        
        # Create visual selection patch (blue)
        plot_dialog.visible_span_patch = patches.Rectangle(
            (start_time, original_ylim[0]),
            end_time - start_time,
            original_ylim[1] - original_ylim[0],  # Use original y-limits
            linewidth=0,
            edgecolor=None,
            facecolor='blue',
            alpha=0.3,
            zorder=1
        )
        ax.add_patch(plot_dialog.visible_span_patch)
        
        # Create cursor lines (blue) using original y-limits
        plot_dialog.cursor_line, = ax.plot([start_time, start_time], original_ylim, 'b', linewidth=1.2)
        if hasattr(plot_dialog, 'spectrogram_ax') and plot_dialog.spectrogram_ax is not None:
            plot_dialog.spectrogram_cursor_line, = plot_dialog.spectrogram_ax.plot(
                [start_time, start_time], original_spec_ylim, 'b', linewidth=1.2)
        
        # Enforce original y-limits
        ax.set_ylim(original_ylim)
        if hasattr(plot_dialog, 'spectrogram_ax') and plot_dialog.spectrogram_ax is not None:
            plot_dialog.spectrogram_ax.set_ylim(original_spec_ylim)
        
        # Prepare audio segment
        start_sample = int(start_time * self.fs)
        end_sample = int(end_time * self.fs)
        segment = audio_signal[start_sample:end_sample].astype(np.float32)
        total_len = len(segment)
        stream_pos = 0
        
        # Save backgrounds for blitting
        canvas = ax.figure.canvas
        canvas.draw()
        plot_dialog._background_waveform = canvas.copy_from_bbox(ax.bbox)
        if hasattr(plot_dialog, 'spectrogram_ax') and plot_dialog.spectrogram_ax is not None:
            try:
                plot_dialog._background_spectrogram = canvas.copy_from_bbox(plot_dialog.spectrogram_ax.bbox)
            except Exception as e:
                print(f"Background save error: {e}")
                plot_dialog._background_spectrogram = None
        
        def update_cursor():
            nonlocal stream_pos
            if not plot_dialog.isVisible():
                if hasattr(plot_dialog, 'cursor_timer'):
                    plot_dialog.cursor_timer.stop()
                return
            
            elapsed = time.time() - t0
            current_time = start_time + elapsed
            
            if current_time >= end_time:
                plot_dialog.cursor_timer.stop()
                return
            
            # Update both cursors using original y-limits
            plot_dialog.cursor_line.set_data([current_time, current_time], original_ylim)
            if hasattr(plot_dialog, 'spectrogram_cursor_line') and plot_dialog.spectrogram_cursor_line is not None:
                plot_dialog.spectrogram_cursor_line.set_data([current_time, current_time], original_spec_ylim)
            
            # Redraw both views
            try:
                canvas.restore_region(plot_dialog._background_waveform)
                ax.draw_artist(plot_dialog.visible_span_patch)
                ax.draw_artist(plot_dialog.cursor_line)
                canvas.blit(ax.bbox)
                
                if (hasattr(plot_dialog, '_background_spectrogram') and 
                   (plot_dialog._background_spectrogram is not None)):
                    canvas.restore_region(plot_dialog._background_spectrogram)
                    plot_dialog.spectrogram_ax.draw_artist(plot_dialog.spectrogram_cursor_line)
                    canvas.blit(plot_dialog.spectrogram_ax.bbox)
            except Exception as e:
                print(f"Drawing error: {e}")
        
        def callback(outdata, frames, time_info, status):
            nonlocal stream_pos
            if status:
                print(status)
            
            remaining = total_len - stream_pos
            if remaining <= 0:
                raise sd.CallbackStop()
            
            n = min(frames, remaining)
            outdata[:n] = segment[stream_pos:stream_pos + n].reshape(-1, 1)
            if n < frames:
                outdata[n:] = 0
            stream_pos += n
        
        # Start playback with fresh timer
        plot_dialog.cursor_timer = QTimer()
        try:
            plot_dialog.active_stream = sd.OutputStream(
                samplerate=self.fs,
                channels=1,
                dtype='float32',
                callback=callback,
                blocksize=1024
            )
            plot_dialog.active_stream.start()
            
            t0 = time.time()
            plot_dialog.cursor_timer.timeout.connect(update_cursor)
            plot_dialog.cursor_timer.start(30)
        except Exception as e:
            print(f"Playback error: {e}")

    def on_plot_window_close(self, plot_id):
        """Centralized cleanup for plot windows"""
        # Stop live analysis if active
        if hasattr(self, 'live_timer') and self.live_timer.isActive():
            self.stop_live_analysis()
        
        # Clean up span selectors - more defensive
        if hasattr(self, 'span_selectors'):
            if plot_id in self.span_selectors:
                for selector in self.span_selectors[plot_id]:
                    try:
                        if hasattr(selector, 'disconnect_events'):
                            selector.disconnect_events()
                        elif hasattr(selector, 'set_active'):
                            selector.set_active(False)  # Alternative cleanup
                    except Exception as e:
                        print(f"Selector cleanup error: {e}")
                self.span_selectors.pop(plot_id, None)
        
        # Clear current figure reference if it matches
        if hasattr(self, 'current_figure') and id(self.current_figure) == plot_id:
            try:
                import matplotlib.pyplot as plt
                plt.close(self.current_figure)
            except:
                pass
            finally:
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

        # Stop all playback and timers if window is closed
        if hasattr(self.parent(), 'stop_audio_playback'):
            self.parent().stop_audio_playback()

        if hasattr(self, 'live_analysis_timer'):
            self.live_analysis_timer.stop()
        
        # Close figures
        if hasattr(self, 'current_figure') and self.current_figure:
            plt.close(self.current_figure)
        
        event.accept()
        super().closeEvent(event)