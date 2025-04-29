from PyQt5.QtWidgets import ( QDialog, QLabel, QPushButton, QLineEdit, QRadioButton, 
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
import parselmouth
import matplotlib as mpl
from pitchAdvancedSettings import AdvancedSettings
from PyQt5.QtWidgets import QVBoxLayout
from scipy.io.wavfile import write
from help import Help
from pathlib import Path


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
        
        # This is the magic flag combination that always works:
        #self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        self.plot_windows = []  # Track all plot windows
        self.selected_span = None  # Track span selection times

        np.seterr(divide='ignore')
        self.span = None
        self.setupUI()

        print("PAST CONSTRUCTOR")

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
            
            if idx_max > idx_min:
                self.selected_span = (xmin, xmax)
                # Update CONTROLLER window title
                self.setWindowTitle(
                    f"{self.base_name}_{self.format_timestamp(xmin)}-{self.format_timestamp(xmax)}"
                )
                # Update all open PLOT windows
                #self.update_plot_window_titles()
                # Play the selected audio
                sd.play(self.audio[idx_min:idx_max], self.fs)
        
        self.span = SpanSelector(
            ax,
            on_select,
            'horizontal',
            useblit=True,
            interactive=True,
            drag_from_anywhere=True
        )

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
        
        # Spectrogram controls
        self.window_type.setEnabled(stft_enabled or spectrogram_enabled or stft_spect_enabled or 
                                  spectral_centroid_enabled or ste_enabled)
        self.window_size.setEnabled(not ft_enabled and not pitch_enabled and not filtering_enabled)
        self.overlap.setEnabled(spectrogram_enabled or stft_spect_enabled or 
                               spectral_centroid_enabled or ste_enabled)
        self.nfft.setEnabled(stft_enabled or spectrogram_enabled or stft_spect_enabled or 
                            spectral_centroid_enabled or ste_enabled)
        self.min_freq.setEnabled(spectrogram_enabled or stft_spect_enabled or spectral_centroid_enabled)
        self.max_freq.setEnabled(spectrogram_enabled or stft_spect_enabled or spectral_centroid_enabled)
        self.draw_style.setEnabled(spectrogram_enabled or stft_spect_enabled or 
                                 spectral_centroid_enabled or filtering_enabled)
        self.show_pitch.setEnabled(spectrogram_enabled)
        
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
            
            self.show_plot_window()

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

    def calculate_pitch(self):
        """Calculate pitch using YIN algorithm with current librosa version"""
        try:
            # Get parameters from UI
            min_pitch = float(self.min_pitch.text())
            max_pitch = float(self.max_pitch.text())
            
            # Convert audio to mono if needed and normalize
            audio = np.mean(self.audio, axis=1) if len(self.audio.shape) > 1 else self.audio
            audio = librosa.util.normalize(audio)
            
            # Calculate pitch using YIN algorithm
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=min_pitch,
                fmax=max_pitch,
                sr=self.fs,
                frame_length=2048,
                hop_length=512,
                fill_na=np.nan  # Fill unvoiced frames with NaN
            )
            
            # Simple smoothing using median filter (replacement for librosa.util.smooth)
            if len(f0) > 0:
                from scipy.ndimage import median_filter
                f0_smoothed = median_filter(f0, size=5)
                f0_smoothed[~voiced_flag] = np.nan
            else:
                f0_smoothed = f0
                
            return f0, f0_smoothed
            
        except Exception as e:
            QMessageBox.warning(self, "Pitch Error", f"Could not calculate pitch: {str(e)}")
            # Return arrays of NaN with appropriate length
            dummy_length = len(self.audio) // 512  # Approximate number of frames
            return np.full(dummy_length, np.nan), np.full(dummy_length, np.nan)
    
    def plot_spectrogram(self):
        try:
            wind_size = float(self.window_size.text())
            overlap = float(self.overlap.text())
            nfft = int(self.nfft.currentText())
            min_freq = int(self.min_freq.text())
            max_freq = int(self.max_freq.text())
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

            wind_size_samples = int(wind_size * self.fs)
            hop_size = wind_size_samples - int(overlap * self.fs)
            window = self.get_window(wind_size_samples)
            
            ax0.plot(time, audio)
            ax0.set(ylabel='Amplitude')
            
            if draw_style == 1:
                D = librosa.stft(audio, n_fft=nfft, hop_length=hop_size,
                                 win_length=wind_size_samples, window=window)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear',
                                               sr=self.fs, hop_length=hop_size,
                                               fmin=min_freq, fmax=max_freq, ax=ax1)
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
            self.annotate_brightest_frequencies(ax1, S_db, hop_size, nfft)

            if show_pitch:
                pitch, pitch_values = self.calculate_pitch()
                pitch_times = librosa.times_like(pitch_values, sr=self.fs, hop_length=hop_size)
                ax1.plot(pitch_times, pitch_values, '-', color='red', linewidth=2, alpha=0.8)

            ax0.set(xlim=[0, time[-1]])
            ax1.set(xlim=[0, time[-1]])

            self.show_plot_window(fig, ax0, audio)  # ← pass individual data
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Spectrogram failed: {str(e)}")

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

    def plot_stft_spect(self):
        """STFT + Spectrogram with interactive window selection"""
        try:
            wind_size = float(self.window_size.text())
            overlap = float(self.overlap.text())
            nfft = int(self.nfft.currentText())
            min_freq = int(self.min_freq.text())
            max_freq = int(self.max_freq.text())
            draw_style = self.draw_style.currentIndex() + 1
            
            # Create figure with adjusted layout - taller spectrogram
            self.current_figure = plt.figure(figsize=(12, 8))  # Increased figure height (from 6 to 8)
            gs = plt.GridSpec(3, 2, width_ratios=[15, 1], height_ratios=[1, 1, 1.5], hspace=0.4)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[1, 0])
            ax3 = plt.subplot(gs[2, 0], sharex=ax1)  # Share x-axis with ax1
            cbar_ax = plt.subplot(gs[:, 1])  # Colorbar uses entire right column
            
            self.current_figure.suptitle('STFT + Spectrogram', y=0.98)  # Adjust title position
            
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
            self.mid_point_idx = len(self.audio) // 2  # Start in middle
            
            # Create initial spectrogram image
            if draw_style == 1:
                D = librosa.stft(self.audio, n_fft=self.nfft_val, hop_length=self.hop_size,
                                win_length=self.wind_size_samples, window=self.window)
                self.S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                self.img = librosa.display.specshow(self.S_db, x_axis='time', y_axis='linear',
                                                sr=self.fs, hop_length=self.hop_size,
                                                fmin=min_freq, fmax=max_freq, ax=ax3)
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
            
            # Create colorbar once on the dedicated axis
            self.cbar = self.current_figure.colorbar(self.img, cax=cbar_ax, format="%+2.0f dB")
            
            # Initial plot
            self.update_stft_spect_plot(ax1, ax2, ax3)
            
            # Set up interactions
            self.span_selector = SpanSelector(
                ax1,
                self.on_span_select,
                'horizontal',
                useblit=True,
                interactive=True,
                drag_from_anywhere=True
            )
            
            self.current_figure.canvas.mpl_connect(
                'button_press_event', 
                lambda e: self.on_window_click_spect(e, ax1, ax2, ax3)
            )
            
            self.show_plot_window(self.current_figure, ax1, self.audio)            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"STFT+Spectrogram plot failed: {str(e)}")

    def on_window_click_spect(self, event, ax1, ax2, ax3):
        """Move analysis window on left click (without dragging)"""
        if event.inaxes != ax1 or event.button != 1 or event.dblclick:
            return
        
        # Only move window on simple click (not drag)
        if hasattr(event, 'pressed') and event.pressed:
            return
            
        # Update window center position
        self.mid_point_idx = np.searchsorted(self.time, event.xdata)
        self.update_stft_spect_plot(ax1, ax2, ax3)

    def update_stft_spect_plot(self, ax1, ax2, ax3):
        """Update plot with proper array handling"""
        # Clear only the plots we need to update (not the spectrogram)
        ax1.clear()
        ax2.clear()
        
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
        
        self.show_plot_window(self.current_figure, ax[0], self.audio)
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
        """Create a span selector for a specific plot window"""
        def onselect(xmin, xmax):
            start_sample = int(xmin * self.fs)
            end_sample = int(xmax * self.fs)
            segment = audio_signal[start_sample:end_sample]
            
            sd.stop()
            sd.play(segment, self.fs)
            
            # Visual feedback just for this plot
            for patch in ax.patches:
                if hasattr(patch, 'get_label') and patch.get_label() == 'selection':
                    patch.remove()
            ax.axvspan(xmin, xmax, color='red', alpha=0.3, label='selection')
            plt.gcf().canvas.draw()
        
        # Remove old selector if exists
        if plot_id in self.span_selectors:
            for selector in self.span_selectors[plot_id]:
                try:
                    selector.disconnect_events()
                except:
                    pass
        
        # Create new selector
        selector = SpanSelector(
            ax,
            onselect,
            'horizontal',
            useblit=True,
            interactive=True,
            drag_from_anywhere=True
        )
        
        # Store selector
        if plot_id not in self.span_selectors:
            self.span_selectors[plot_id] = []
        self.span_selectors[plot_id].append(selector)

    def plot_filtered_spectrogram(self, filter_type, filtered_signal):
        """Plot original and filtered spectrograms with proper span selectors"""
        try:
            self.current_figure = plt.figure(figsize=(12, 8))
            gs = plt.GridSpec(2, 1, height_ratios=[1, 1])
            
            # Get unique identifier for this plot window
            plot_id = id(self.current_figure.canvas.manager.window)
            
            # Original spectrogram
            ax0 = plt.subplot(gs[0])
            D_orig = librosa.stft(self.audio, n_fft=2048, hop_length=512)
            S_db_orig = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)
            librosa.display.specshow(S_db_orig, x_axis='time', y_axis='log',
                                   sr=self.fs, ax=ax0)
            ax0.set(title='Original Signal Spectrogram')
            self.create_span_selector(ax0, self.audio, plot_id)  # Pass plot_id
            
            # Filtered spectrogram
            ax1 = plt.subplot(gs[1])
            D_filt = librosa.stft(filtered_signal, n_fft=2048, hop_length=512)
            S_db_filt = librosa.amplitude_to_db(np.abs(D_filt), ref=np.max)
            librosa.display.specshow(S_db_filt, x_axis='time', y_axis='log',
                                   sr=self.fs, ax=ax1)
            ax1.set(title=f'Filtered Signal Spectrogram ({filter_type})')
            self.create_span_selector(ax1, filtered_signal, plot_id)  # Pass plot_id
            
            self.current_figure.tight_layout()
            self.show_plot_window(self.current_figure, ax1, self.audio)
            
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
        """Clean up resources when plot window closes"""
        if plot_id in self.span_selectors:
            for selector in self.span_selectors[plot_id]:
                try:
                    selector.disconnect_events()
                except:
                    pass
            del self.span_selectors[plot_id]
        
        self.plot_windows = [w for w in self.plot_windows 
                            if hasattr(w, 'plot_id') and w.plot_id != plot_id]


    def show_plot_window(self, figure, waveform_ax, audio_signal):
        method = self.method_selector.currentText()
        start, end = 0, self.duration
        plot_title = f"{method}_{self.base_name}_{self.format_timestamp(start)}-{self.format_timestamp(end)}"

        plot_dialog = QDialog(self)
        plot_dialog.setWindowTitle(plot_title)
        plot_dialog.setAttribute(Qt.WA_DeleteOnClose)
        plot_dialog.plot_id = id(plot_dialog)

        layout = QVBoxLayout()
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, plot_dialog)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        plot_dialog.setLayout(layout)

        self.create_span_selector(waveform_ax, audio_signal, plot_dialog.plot_id)  # ← pass correct refs

        plot_dialog.destroyed.connect(lambda: self.cleanup_plot_window(plot_dialog.plot_id))
        self.plot_windows.append(plot_dialog)
        plot_dialog.show()



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
        
        # Close figures
        if hasattr(self, 'current_figure') and self.current_figure:
            plt.close(self.current_figure)
        
        event.accept()



