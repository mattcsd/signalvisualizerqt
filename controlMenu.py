from PyQt5.QtWidgets import (QDialog, QLabel, QPushButton, QLineEdit, QRadioButton, 
                            QCheckBox, QComboBox, QGridLayout, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt
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
import parselmouth
import matplotlib as mpl
from pitchAdvancedSettings import AdvancedSettings
from PyQt5.QtWidgets import QVBoxLayout
from scipy.io.wavfile import write


class ControlMenu(QDialog):
    def __init__(self, name, fs, audio, duration, controller):
        super().__init__()
        self.fileName = name
        self.audio = audio
        self.fs = fs
        self.duration = duration
        self.lenAudio = len(audio)
        self.time = np.arange(0, self.lenAudio/self.fs, 1/self.fs)
        self.controller = controller
        self.current_figure = None
        
        np.seterr(divide='ignore')
        self.span = None
        self.setupUI()
        
    def setupUI(self):
        self.setWindowTitle(self.fileName)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        main_layout = QGridLayout()
        main_layout.setVerticalSpacing(8)
        main_layout.setHorizontalSpacing(10)
        
        # Analysis method selector - placed above all groups
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
        self.create_ste_group(main_layout)
        
        self.plot_button = QPushButton('Plot')
        self.plot_button.clicked.connect(self.plot_figure)
        main_layout.addWidget(self.plot_button, 14, 3, 1, 1)
        
        self.help_button = QPushButton('🛈')
        self.help_button.setFixedWidth(30)
        self.help_button.clicked.connect(lambda: self.controller.help.createHelpMenu(8))
        main_layout.addWidget(self.help_button, 14, 2, 1, 1, Qt.AlignRight)
        
        self.setLayout(main_layout)
        self.update_ui_state('Spectrogram')

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
        """Enable/disable pitch controls based on checkbox state"""
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
        
        group.setLayout(grid)
        layout.addWidget(group, 1, 2, 7, 2)
        
    def update_filter_ui(self, filter_type):
        """Update filter UI based on selected filter type"""
        harmonic = filter_type == 'Harmonic'
        lphp = filter_type in ['Lowpass', 'Highpass']
        bpbs = filter_type in ['Bandpass', 'Bandstop']
        
        self.fund_freq.setEnabled(harmonic)
        self.center_freq.setEnabled(harmonic)
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
        layout.addWidget(group, 8, 2, 6, 2)
        
    def update_ui_state(self, method):
        """Update UI state based on selected analysis method"""
        # Enable/disable controls according to original Tkinter logic
        self.window_type.setEnabled(method in ['STFT', 'Spectrogram', 'STFT + Spect', 'Spectral Centroid', 'Short-Time-Energy'])
        self.window_size.setEnabled(method not in ['FT', 'Pitch', 'Filtering'])
        self.overlap.setEnabled(method in ['Spectrogram', 'STFT + Spect', 'Spectral Centroid', 'Short-Time-Energy'])
        self.nfft.setEnabled(method in ['STFT', 'Spectrogram', 'STFT + Spect', 'Spectral Centroid'])
        self.min_freq.setEnabled(method in ['Spectrogram', 'STFT + Spect', 'Spectral Centroid'])
        self.max_freq.setEnabled(method in ['Spectrogram', 'STFT + Spect', 'Spectral Centroid'])
        self.draw_style.setEnabled(method in ['Spectrogram', 'STFT + Spect', 'Spectral Centroid', 'Filtering'])
        self.show_pitch.setEnabled(method == 'Spectrogram')
        
        # Pitch controls
        pitch_active = method == 'Pitch'
        self.pitch_method.setEnabled(pitch_active)
        self.min_pitch.setEnabled(pitch_active)
        self.max_pitch.setEnabled(pitch_active)
        self.adv_settings.setEnabled(pitch_active)
        
        # Filter controls
        filter_active = method == 'Filtering'
        self.filter_type.setEnabled(filter_active)
        if filter_active:
            self.update_filter_ui(self.filter_type.currentText())
        
        # STE controls
        ste_active = method == 'Short-Time-Energy'
        self.beta.setEnabled(ste_active and self.window_type.currentText() == 'Kaiser')
        


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
        
        self.show_plot_window()

    def plot_stft(self):
        """STFT with proper array dimension handling"""
        try:
            wind_size = float(self.window_size.text())
            nfft = int(self.nfft.currentText())
            
            self.current_figure, ax = plt.subplots(2, figsize=(12,6))
            self.current_figure.suptitle('STFT Analysis')
            
            # Ensure time and audio arrays match
            if len(self.time) > len(self.audio):
                self.time = self.time[:len(self.audio)]
            elif len(self.audio) > len(self.time):
                self.audio = self.audio[:len(self.time)]
            
            # STFT analysis properties
            self.wind_size_samples = int(wind_size * self.fs)
            self.window = self.get_window(self.wind_size_samples)
            self.nfft_val = nfft
            self.mid_point_idx = len(self.audio) // 2  # Start in middle
            
            # Initial plot
            self.update_stft_plot(ax)
            
            # Set up interactions
            self.span_selector = SpanSelector(
                ax[0],
                self.on_span_select,
                'horizontal',
                useblit=True,
                interactive=True,
                drag_from_anywhere=True
            )
            
            self.current_figure.canvas.mpl_connect(
                'button_press_event', 
                lambda e: self.on_window_click(e, ax)
            )
            
            self.show_plot_window()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"STFT plot failed: {str(e)}")

    def on_window_click(self, event, ax):
        """Move analysis window on left click (without dragging)"""
        if event.inaxes != ax[0] or event.button != 1 or event.dblclick:
            return
        
        # Only move window on simple click (not drag)
        if hasattr(event, 'pressed') and event.pressed:
            return
            
        # Update window center position
        self.mid_point_idx = np.searchsorted(self.time, event.xdata)
        self.update_stft_plot(ax)

    def on_span_select(self, xmin, xmax):
        """Handle span selection for playback (on drag)"""
        start = np.searchsorted(self.time, xmin)
        end = np.searchsorted(self.time, xmax)
        segment = self.audio[start:end]
        sd.play(segment, self.fs)
        
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

    def plot_spectrogram(self):
        try:
            wind_size = float(self.window_size.text())
            overlap = float(self.overlap.text())
            nfft = int(self.nfft.currentText())
            min_freq = int(self.min_freq.text())
            max_freq = int(self.max_freq.text())
            draw_style = self.draw_style.currentIndex() + 1
            show_pitch = self.show_pitch.isChecked()
            
            self.current_figure = plt.figure(figsize=(12,6))
            gs = self.current_figure.add_gridspec(2, hspace=0, height_ratios=[1, 3])
            ax = gs.subplots(sharex=True)
            self.current_figure.suptitle('Spectrogram')
            
            wind_size_samples = int(wind_size * self.fs)
            hop_size = wind_size_samples - int(overlap * self.fs)
            window = self.get_window(wind_size_samples)
            
            if draw_style == 1:
                D = librosa.stft(self.audio, n_fft=nfft, hop_length=hop_size, 
                                win_length=wind_size_samples, window=window)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear',
                                            sr=self.fs, hop_length=hop_size, 
                                            fmin=min_freq, fmax=max_freq, ax=ax[1])
            else:
                S = librosa.feature.melspectrogram(y=self.audio, sr=self.fs, 
                                                n_fft=nfft, hop_length=hop_size,
                                                win_length=wind_size_samples, 
                                                window=window, fmin=min_freq, 
                                                fmax=max_freq)
                S_db = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                            sr=self.fs, hop_length=hop_size,
                                            fmin=min_freq, fmax=max_freq, ax=ax[1])
            
            self.current_figure.colorbar(img, ax=ax[1], format="%+2.0f dB")
            
            if show_pitch:
                pitch, pitch_values = self.calculate_pitch()
                ax[1].plot(pitch.xs(), pitch_values, '-', color='white')
            
            # Set matching x-axis limits for both subplots
            ax[0].plot(self.time, self.audio)
            ax[0].set(xlim=[0, self.duration], ylabel='Amplitude')
            ax[1].set(xlim=[0, self.duration])  # Explicitly set spectrogram limits
            
            self.show_plot_window()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Spectrogram failed: {str(e)}")

    def plot_stft_spect(self):
        wind_size = float(self.window_size.text())
        overlap = float(self.overlap.text())
        nfft = int(self.nfft.currentText())
        min_freq = int(self.min_freq.text())
        max_freq = int(self.max_freq.text())
        draw_style = self.draw_style.currentIndex() + 1
        
        self.current_figure = plt.figure(figsize=(12,6))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313, sharex=ax1)
        self.current_figure.suptitle('STFT + Spectrogram')
        
        wind_size_samples = int(wind_size * self.fs)
        hop_size = wind_size_samples - int(overlap * self.fs)
        window = self.get_window(wind_size_samples)
        
        audio_segment, (ini, end) = self.get_middle_segment(wind_size_samples)
        audio_segment = audio_segment * window
        
        stft = np.fft.fft(audio_segment, nfft)[:int(nfft/2)]
        freqs = np.arange(int(nfft/2)) * self.fs / nfft
        
        ax1.plot(self.time, self.audio)
        ax1.axvspan(ini, end, color='silver', alpha=0.5)
        
        ax2.plot(freqs, 20*np.log10(abs(stft)))
        ax2.set(xlim=[0, max(freqs)], xlabel='Frequency (Hz)', ylabel='Amplitude (dB)')
        
        if draw_style == 1:
            D = librosa.stft(self.audio, n_fft=nfft, hop_length=hop_size, 
                            win_length=wind_size_samples, window=window)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear',
                                        sr=self.fs, hop_length=hop_size, 
                                        fmin=min_freq, fmax=max_freq, ax=ax3)
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
        
        self.current_figure.colorbar(img, ax=ax3, format="%+2.0f dB")
        
        self.show_plot_window()

    def plot_ste(self):
        wind_size = float(self.window_size.text())
        overlap = float(self.overlap.text())
        window_type = self.window_type.currentText()
        beta = float(self.beta.text()) if window_type == 'Kaiser' else 0
        
        self.current_figure, ax = plt.subplots(2, figsize=(12,6))
        self.current_figure.suptitle('Short-Time Energy')
        
        wind_size_samples = int(wind_size * self.fs)
        hop_size = wind_size_samples - int(overlap * self.fs)
        
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
        
        ste = np.zeros(len(self.audio))
        for i in range(0, len(self.audio) - wind_size_samples, hop_size):
            segment = self.audio[i:i+wind_size_samples] * window
            ste[i:i+wind_size_samples] += segment**2
        
        ax[0].plot(self.time, self.audio)
        ax[1].plot(self.time, ste)
        ax[1].set(xlim=[0, self.duration], xlabel='Time (s)', ylabel='Energy')
        
        self.show_plot_window()

    def plot_pitch(self):
        method = self.pitch_method.currentText()
        min_pitch = float(self.min_pitch.text())
        max_pitch = float(self.max_pitch.text())
        
        self.current_figure, ax = plt.subplots(2, figsize=(12,6))
        self.current_figure.suptitle('Pitch Contour')
        
        write('temp_pitch.wav', self.fs, self.audio)
        snd = parselmouth.Sound('temp_pitch.wav')
        
        if method == 'Autocorrelation':
            pitch = snd.to_pitch_ac(pitch_floor=min_pitch, pitch_ceiling=max_pitch)
        elif method == 'Cross-correlation':
            pitch = snd.to_pitch_cc(pitch_floor=min_pitch, pitch_ceiling=max_pitch)
        elif method == 'Subharmonics':
            pitch = snd.to_pitch_shs(minimum_pitch=min_pitch, ceiling=max_pitch)
        elif method == 'Spinet':
            pitch = snd.to_pitch_spinet(ceiling=max_pitch)
        
        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values == 0] = np.nan
        
        ax[0].plot(self.time, self.audio)
        ax[1].plot(pitch.xs(), pitch_values, '-')
        ax[1].set(xlim=[0, self.duration], ylim=[min_pitch, max_pitch], 
                 xlabel='Time (s)', ylabel='Frequency (Hz)')
        
        self.show_plot_window()

    def plot_spectral_centroid(self):
        wind_size = float(self.window_size.text())
        overlap = float(self.overlap.text())
        nfft = int(self.nfft.currentText())
        min_freq = int(self.min_freq.text())
        max_freq = int(self.max_freq.text())
        draw_style = self.draw_style.currentIndex() + 1
        
        self.current_figure = plt.figure(figsize=(12,6))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313, sharex=ax1)
        self.current_figure.suptitle('Spectral Centroid')
        
        wind_size_samples = int(wind_size * self.fs)
        hop_size = wind_size_samples - int(overlap * self.fs)
        window = self.get_window(wind_size_samples)
        
        audio_segment, (ini, end) = self.get_middle_segment(wind_size_samples)
        audio_segment = audio_segment * window
        
        spectral_centroid = self.calculate_sc(audio_segment)
        sc_value = f"{spectral_centroid:.2f}"
        
        ax1.plot(self.time, self.audio)
        ax1.axvspan(ini, end, color='silver', alpha=0.5)
        
        _, freqs = ax2.psd(audio_segment, NFFT=wind_size_samples, Fs=self.fs, 
                          window=window, noverlap=int(overlap * self.fs))
        ax2.axvline(x=spectral_centroid, color='r')
        ax2.set(xlim=[0, max(freqs)], title=f'Spectral Centroid: {sc_value} Hz')
        
        if draw_style == 1:
            D = librosa.stft(self.audio, n_fft=nfft, hop_length=hop_size, 
                            win_length=wind_size_samples, window=window)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear',
                                        sr=self.fs, hop_length=hop_size, 
                                        fmin=min_freq, fmax=max_freq, ax=ax3)
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
        
        sc = librosa.feature.spectral_centroid(y=self.audio, sr=self.fs, 
                                             n_fft=nfft, hop_length=hop_size,
                                             win_length=wind_size_samples)
        times = librosa.times_like(sc, sr=self.fs, hop_length=hop_size)
        ax3.plot(times, sc.T, color='w')
        
        self.current_figure.colorbar(img, ax=ax3, format="%+2.0f dB")
        
        self.show_plot_window()

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
        
        ax[0].plot(self.time, self.audio)
        ax[0].set(xlim=[0, self.duration], title='Original Signal')
        
        ax[1].plot(self.time, filtered_signal)
        ax[1].set(xlim=[0, self.duration], title='Filtered Signal')
        
        self.show_plot_window()

    def calculate_sc(self, segment):
        magnitudes = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), 1/self.fs)
        return np.sum(magnitudes * freqs) / np.sum(magnitudes)

    def show_plot_window(self):
        if not self.current_figure:
            return
            
        plot_dialog = QDialog()
        plot_dialog.setWindowTitle(self.fileName)
        layout = QVBoxLayout()
        
        canvas = FigureCanvas(self.current_figure)
        toolbar = NavigationToolbar(canvas, plot_dialog)
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        plot_dialog.setLayout(layout)
        
        if len(self.current_figure.axes) > 0:
            self.createSpanSelector(self.current_figure.axes[0])
        
        plot_dialog.exec_()

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

    def createSpanSelector(self, ax):
        if self.span is not None:
            try:
                self.span.disconnect_events()
            except:
                pass
            self.span = None
        
        def on_select(xmin, xmax):
            idx_min = np.argmax(self.time >= xmin)
            idx_max = np.argmax(self.time >= xmax)
            selected_audio = self.audio[idx_min:idx_max]
            sd.play(selected_audio, self.fs)
            
        self.span = SpanSelector(
            ax,
            on_select,
            'horizontal',
            useblit=True,
            interactive=True,
            drag_from_anywhere=True
        )



    def closeEvent(self, event):
        if self.current_figure:
            plt.close(self.current_figure)
        plt.close('all')
        event.accept()