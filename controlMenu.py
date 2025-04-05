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
        
        # Ignore divide by zero warnings
        np.seterr(divide='ignore')
        self.span = None  # Initialize span selector attribute
        self.setupUI()
        
    def setupUI(self):
        self.setWindowTitle(self.fileName)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        # Main layout
        main_layout = QGridLayout()
        main_layout.setVerticalSpacing(8)
        main_layout.setHorizontalSpacing(10)
        
        # Create groups
        self.create_spectrogram_group(main_layout)
        self.create_pitch_group(main_layout)
        self.create_filter_group(main_layout)
        self.create_ste_group(main_layout)
        
        # Add plot button
        self.plot_button = QPushButton('Plot')
        self.plot_button.clicked.connect(self.plot_figure)
        main_layout.addWidget(self.plot_button, 14, 3, 1, 1)
        
        # Add help button
        self.help_button = QPushButton('🛈')
        self.help_button.setFixedWidth(30)
        self.help_button.clicked.connect(lambda: self.controller.help.createHelpMenu(8))
        main_layout.addWidget(self.help_button, 14, 2, 1, 1, Qt.AlignRight)
        
        self.setLayout(main_layout)
        
    def create_spectrogram_group(self, layout):
        group = QGroupBox("Spectrogram")
        grid = QGridLayout()
        
        # Window type
        self.window_type = QComboBox()
        self.window_type.addItems(['Bartlett', 'Blackman', 'Hamming', 'Hanning', 'Kaiser'])
        
        # Window size
        self.window_size = QLineEdit()
        self.window_size.setValidator(QDoubleValidator(0.01, self.duration, 3))
        if self.duration <= 0.03:
            self.window_size.setText(f"{self.duration - 0.001:.3f}")
        else:
            self.window_size.setText("0.03")
            
        # Overlap
        self.overlap = QLineEdit()
        self.overlap.setValidator(QDoubleValidator(0.0, self.duration, 3))
        if self.duration <= 0.03:
            wind_size = float(self.window_size.text())
            self.overlap.setText(f"{wind_size - 0.001:.3f}")
        else:
            self.overlap.setText("0.01")
            
        # NFFT
        self.nfft = QComboBox()
        nfft_values = [2**i for i in range(9, 20)]
        self.nfft.addItems(map(str, nfft_values))
        
        # Frequency range
        self.min_freq = QLineEdit("0")
        self.max_freq = QLineEdit(str(self.fs//2))
        
        # Drawing style
        self.draw_style = QComboBox()
        self.draw_style.addItems(["Linear", "Mel"])
        
        # Pitch checkbox
        self.show_pitch = QCheckBox("Show Pitch")
        
        # Add widgets to grid
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
        layout.addWidget(group, 0, 0, 8, 2)
        
    def create_pitch_group(self, layout):
        group = QGroupBox("Pitch")
        grid = QGridLayout()
        
        # Method
        self.pitch_method = QComboBox()
        self.pitch_method.addItems(['Autocorrelation', 'Cross-correlation', 'Subharmonics', 'Spinet'])
        
        # Pitch range
        self.min_pitch = QLineEdit("75.0")
        self.max_pitch = QLineEdit("600.0")
        
        # Advanced settings button
        self.adv_settings = QPushButton("Advanced Settings")
        self.adv_settings.clicked.connect(self.controller.adse.advancedSettings)        
       
        # Add widgets to grid
        grid.addWidget(QLabel("Method:"), 0, 0)
        grid.addWidget(self.pitch_method, 0, 1)
        grid.addWidget(QLabel("Min pitch (Hz):"), 1, 0)
        grid.addWidget(self.min_pitch, 1, 1)
        grid.addWidget(QLabel("Max pitch (Hz):"), 2, 0)
        grid.addWidget(self.max_pitch, 2, 1)
        grid.addWidget(self.adv_settings, 3, 0, 1, 2)
        
        group.setLayout(grid)
        layout.addWidget(group, 8, 0, 5, 2)

    def show_advanced_settings(self):

        dialog = AdvancedSettings(self)
        if dialog.exec_() == QDialog.Accepted:
            autocorr_vars = dialog.getAutocorrelationVars()
            subharmonic_vars = dialog.getSubharmonicsVars()
            spinet_vars = dialog.getSpinetVars()
            other_vars = dialog.getVariables()

            print("Autocorrelation Vars:", autocorr_vars)
            print("Subharmonics Vars:", subharmonic_vars)
            print("Spinet Vars:", spinet_vars)
            print("Other Vars:", other_vars)

            # Store in controller
            self.controller.adse = {
                'autocorr': autocorr_vars,
                'subharmonics': subharmonic_vars,
                'spinet': spinet_vars,
                'other': other_vars
            }

            
    def create_filter_group(self, layout):
        group = QGroupBox("Filtering")
        grid = QGridLayout()
        
        # Filter type
        self.filter_type = QComboBox()
        self.filter_type.addItems(['Harmonic', 'Lowpass', 'Highpass', 'Bandpass', 'Bandstop'])
        
        # Filter parameters
        self.fund_freq = QLineEdit("1")
        self.center_freq = QLineEdit("400")
        self.percentage = QLineEdit("10.0")
        self.fcut = QLineEdit("1000")
        self.fcut1 = QLineEdit("200")
        self.fcut2 = QLineEdit("600")
        
        # Add widgets to grid
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
        layout.addWidget(group, 0, 2, 7, 2)
        
    def create_ste_group(self, layout):
        group = QGroupBox("Short-Time Energy")
        grid = QGridLayout()
        
        # Beta parameter
        self.beta = QLineEdit("0")
        
        # Add widgets to grid
        grid.addWidget(QLabel("Beta:"), 0, 0)
        grid.addWidget(self.beta, 0, 1)
        
        group.setLayout(grid)
        layout.addWidget(group, 7, 2, 6, 2)
        
    def plot_figure(self):
        try:
            # Validate inputs
            wind_size = float(self.window_size.text())
            overlap = float(self.overlap.text())
            min_freq = int(self.min_freq.text())
            max_freq = int(self.max_freq.text())
            
            if wind_size > self.duration or wind_size <= 0:
                QMessageBox.warning(self, "Error", "Window size must be positive and <= duration")
                return
                
            if overlap >= wind_size:
                QMessageBox.warning(self, "Error", "Overlap must be < window size")
                return
                
            if min_freq >= max_freq:
                QMessageBox.warning(self, "Error", "Min freq must be < max freq")
                return
                
            if max_freq > self.fs//2:
                QMessageBox.warning(self, "Error", f"Max freq must be <= {self.fs//2}")
                return
                
            # Get other parameters
            window_type = self.window_type.currentText()
            nfft = int(self.nfft.currentText())
            draw_style = self.draw_style.currentIndex() + 1  # 1=linear, 2=mel
            show_pitch = self.show_pitch.isChecked()
            
            # Plot spectrogram
            self.plot_spectrogram(window_type, wind_size, overlap, nfft, 
                                min_freq, max_freq, draw_style, show_pitch)
            
        except ValueError as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {str(e)}")

            
    def createSpanSelector(self, ax):
        """Create a span selector for audio playback on the given axis."""
        # Remove existing span selector if it exists
        if self.span is not None:
            try:
                self.span.disconnect_events()
            except:
                pass
            self.span = None
        
        def on_select(xmin, xmax):
            if len(self.audio) <= 1:
                return
                
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
            
    def plot_spectrogram(self, window_type, wind_size, overlap, nfft, 
                    min_freq, max_freq, draw_style, show_pitch):
        try:
            # Create new figure
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            
            # Create plot window
            self.plot_window = QDialog()
            self.plot_window.setWindowTitle(f"Spectrogram - {self.fileName}")
            layout = QVBoxLayout()
            
            # Create canvas and add to layout
            self.canvas = FigureCanvas(self.fig)
            self.toolbar = NavigationToolbar(self.canvas, self.plot_window)
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)
            self.plot_window.setLayout(layout)
            
            # Calculate spectrogram
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
                window = np.kaiser(wind_size_samples, float(self.beta.text()))
            
            if draw_style == 1:  # Linear
                D = librosa.stft(self.audio, n_fft=nfft, hop_length=hop_size, 
                                win_length=wind_size_samples, window=window)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear',
                                            sr=self.fs, hop_length=hop_size, 
                                            fmin=min_freq, fmax=max_freq, ax=self.ax)
            else:  # Mel
                S = librosa.feature.melspectrogram(y=self.audio, sr=self.fs, 
                                                n_fft=nfft, hop_length=hop_size,
                                                win_length=wind_size_samples, 
                                                window=window, fmin=min_freq, 
                                                fmax=max_freq)
                S_db = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                            sr=self.fs, hop_length=hop_size,
                                            fmin=min_freq, fmax=max_freq, ax=self.ax)
            
            # Add colorbar
            self.fig.colorbar(img, ax=self.ax, format="%+2.0f dB")
            
            # Add pitch if requested
            if show_pitch:
                pitch, pitch_values = self.calculate_pitch()
                self.ax.plot(pitch.xs(), pitch_values, '-', color='white')
            
            # Create span selector directly (no separate method needed)
            def onselect(xmin, xmax):
                idx_min = np.argmax(self.time >= xmin)
                idx_max = np.argmax(self.time >= xmax)
                selected_audio = self.audio[idx_min:idx_max]
                sd.play(selected_audio, self.fs)
                
            self.span = SpanSelector(
                self.ax,
                onselect,
                'horizontal',
                useblit=True,
                interactive=True,
                drag_from_anywhere=True
            )
            
            # Show the window
            self.canvas.draw()
            self.plot_window.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create plot: {str(e)}")
            if hasattr(self, 'fig'):
                plt.close(self.fig)
        
    def calculate_pitch(self):
        method = self.pitch_method.currentText()
        min_pitch = float(self.min_pitch.text())
        max_pitch = float(self.max_pitch.text())
        
        # Write temporary file
        write('temp.wav', self.fs, self.audio)
        snd = parselmouth.Sound('temp.wav')
        
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
        
        return pitch, pitch_values
        
    def closeEvent(self, event):
        plt.close('all')
        event.accept()