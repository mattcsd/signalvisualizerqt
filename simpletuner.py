import numpy as np
import pyaudio
from scipy.fft import fft
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QCheckBox,QWidget, QSlider, QVBoxLayout, QComboBox, QLabel, 
                            QHBoxLayout, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class AudioFFTVisualizer(QWidget):
    def __init__(self, master, controller):
        super().__init__(master)
        
        # Audio parameters
        self.CHUNK = 2048 * 4  # Samples per buffer
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.frequency_range = (20, 20000)  # Human hearing range
        
        # Initialize PyAudio and get devices
        self.p = pyaudio.PyAudio()
        self.input_devices = self.get_input_devices()
        self.current_device_index = None  # Will be set when starting stream

        # Visualization parameters
        self.fft_max_scale = 1.0  # Initial fixed scale for FFT Y-axis
        self.fft_min_scale = 0.0   # Minimum Y-axis value (fixed at 0)
        
        # List audio devices
        #self.list_audio_devices()

        # Setup matplotlib figure and canvas
        self.setup_ui()

        # Start audio stream
        self.start_audio_stream()
        
        # Data buffers
        self.audio_data = np.zeros(self.CHUNK)
        self.fft_data = np.zeros(self.CHUNK//2)
        self.scale = 1.0
        self.running = True
        
        # Timer for updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(20)  # Update every 20ms

    def get_input_devices(self):
        """Get list of available input devices with their indices"""
        devices = []
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['maxInputChannels']
                })
        return devices

    def setup_ui(self):
        """Configure the main UI layout with proper Qt widgets"""
        main_layout = QVBoxLayout(self)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Create control panel with Qt widgets
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Device selection dropdown
        self.device_label = QLabel("Input Device:")
        self.device_dropdown = QComboBox()
        self.populate_device_dropdown()
        self.device_dropdown.currentIndexChanged.connect(self.change_device)
        
        # Scale controls
        self.scale_label = QLabel("FFT Scale:")
        self.scale_slider = QSlider()
        self.scale_slider.setOrientation(Qt.Horizontal)
        self.scale_slider.setRange(10, 50)  # 0.1 to 5.0 in steps of 0.1
        self.scale_slider.setValue(10)  # Default 1.0
        self.scale_slider.valueChanged.connect(self.update_qt_scale)
        
        # Y-axis max controls
        self.ymax_label = QLabel("Y-axis Max:")
        self.ymax_slider = QSlider()
        self.ymax_slider.setOrientation(Qt.Horizontal)
        self.ymax_slider.setRange(10, 50)  # 0.1 to 5.0 in steps of 0.1
        self.ymax_slider.setValue(10)  # Default 1.0
        self.ymax_slider.valueChanged.connect(self.update_qt_ymax)

        # Create checkbox for log/linear frequency scale
        self.log_freq_checkbox = QCheckBox("Log Frequency Axis")
        self.log_freq_checkbox.setChecked(False)  # Default to linear

        # Optional: connect it to trigger redraw when toggled
        self.log_freq_checkbox.stateChanged.connect(self.update_plot)

        # Add it to your layout (adjust for your layout variable)
        control_layout.addWidget(self.log_freq_checkbox)


        
        # Add widgets to control panel
        control_layout.addWidget(self.device_label)
        control_layout.addWidget(self.device_dropdown)
        control_layout.addWidget(self.scale_label)
        control_layout.addWidget(self.scale_slider)
        control_layout.addWidget(self.ymax_label)
        control_layout.addWidget(self.ymax_slider)
        
        # Add stretch to push controls left
        control_layout.addStretch()
        
        # Add widgets to main layout
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(control_panel)
        
        # Setup plots
        self.setup_plots()

    def populate_device_dropdown(self):
        """Populate the dropdown with available devices"""
        self.device_dropdown.clear()
        for dev in self.input_devices:
            self.device_dropdown.addItem(
                f"{dev['name']} (Ch: {dev['channels']})", 
                userData=dev['index']
            )
        
        # Select default device if available
        default_device = self.p.get_default_input_device_info()
        if default_device:
            for i, dev in enumerate(self.input_devices):
                if dev['index'] == default_device['index']:
                    self.device_dropdown.setCurrentIndex(i)
                    break

    def setup_plots(self):
        """Configure matplotlib plots"""
        # Create subplots
        self.ax_wave = self.figure.add_subplot(211)
        self.ax_fft = self.figure.add_subplot(212)
        
        # Waveform plot
        self.x_wave = np.arange(0, self.CHUNK)
        self.line_wave, = self.ax_wave.plot(self.x_wave, np.zeros(self.CHUNK), 'b')
        self.ax_wave.set_title('Time Domain - Microphone Input')
        self.ax_wave.set_xlim(0, self.CHUNK)
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.set_xlabel('Samples')
        self.ax_wave.set_ylabel('Amplitude')
        self.ax_wave.grid(True)
        
        # FFT plot with fixed Y-axis
        self.x_fft = np.linspace(0, self.RATE/2, self.CHUNK//2)
        self.line_fft, = self.ax_fft.semilogx(self.x_fft, np.zeros(self.CHUNK//2), 'r')
        self.ax_fft.set_title('Frequency Domain - FFT Analysis')
        self.ax_fft.set_xlim(*self.frequency_range)
        self.ax_fft.set_ylim(self.fft_min_scale, self.fft_max_scale)
        self.ax_fft.set_xlabel('Frequency (Hz)')
        self.ax_fft.set_ylabel('Magnitude')
        self.ax_fft.grid(True, which='both')

    def add_controls(self):
        """Add UI controls to the plot"""
        axcolor = 'lightgoldenrodyellow'
        
        # FFT Scale slider (controls the data scaling)
        self.ax_scale = self.figure.add_axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)
        self.scale_slider = widgets.Slider(
            self.ax_scale, 'Data Scale', 0.1, 5.0, valinit=1.0, valstep=0.1
        )
        self.scale_slider.on_changed(self.update_scale)
        
        # Y-axis Max slider (controls the fixed Y-axis limit)
        self.ax_ymax = self.figure.add_axes([0.2, 0.06, 0.6, 0.03], facecolor=axcolor)
        self.ymax_slider = widgets.Slider(
            self.ax_ymax, 'Y-axis Max', 0.1, 5.0, valinit=self.fft_max_scale, valstep=0.1
        )
        self.ymax_slider.on_changed(self.update_ymax)
        
        # Device selection dropdown
        self.ax_device = self.figure.add_axes([0.2, 0.02, 0.6, 0.03], facecolor=axcolor)
        
        # Create device labels for dropdown
        device_labels = [f"{dev['index']}: {dev['name']} (Ch: {dev['channels']})" 
                        for dev in self.input_devices]
        
        # Create dropdown (RadioButtons styled as dropdown)
        self.device_selector = widgets.RadioButtons(
            self.ax_device,
            labels=device_labels,
            active=0  # Default to first device
        )
        
        # Set smaller font size if many devices
        if len(device_labels) > 3:
            for label in self.device_selector.labels:
                label.set_fontsize(8)
        
        self.device_selector.on_clicked(self.change_device)

    def change_device(self, index):
        """Handle device selection change"""
        device_index = self.device_dropdown.itemData(index)
        print(f"Selected device index: {device_index}")
        self.start_audio_stream(device_index)

    def update_qt_scale(self, value):
        """Update dB vertical shift from Qt slider"""
        self.scale = value / 10.0  # Slider 10–50 → offset -20 dB to +80 dB


    def update_qt_ymax(self, value):
        """Update Y-axis max from Qt slider"""
        self.fft_max_scale = value / 10.0  # Convert from 10-50 to 1.0-5.0
        self.ax_fft.set_ylim(self.fft_min_scale, self.fft_max_scale)
        self.canvas.draw()

    def update_ymax(self, val):
        """Update the fixed Y-axis maximum value"""
        self.fft_max_scale = val
        self.ax_fft.set_ylim(self.fft_min_scale, self.fft_max_scale)
        self.canvas.draw()


    def start_audio_stream(self, device_index=None):
        """Start the audio input stream"""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        try:
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                output=False,
                frames_per_buffer=self.CHUNK,
                input_device_index=device_index,
                stream_callback=self.audio_callback
            )
            self.current_device_index = device_index
            print(f"Stream started with device index: {device_index if device_index else 'default'}")
        except Exception as e:
            print(f"Error opening stream: {e}")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if self.running:
            self.audio_data = np.frombuffer(in_data, dtype=np.int16) / 32768.0
        return (in_data, pyaudio.paContinue)

    def update_scale(self, val):
        """Update the FFT scale factor"""
        self.scale = val

    def change_device(self, event):
        """Change the audio input device"""
        device_index = int(input("Enter device index: "))
        self.start_audio_stream(device_index)

    def update_plot(self):
        """Update the plots with new audio data"""
        if not self.running or not hasattr(self, 'audio_data'):
            return

        # ── Update waveform plot ─────────────────────────────────────
        self.line_wave.set_ydata(self.audio_data)

        # ── Compute FFT with Hann window ─────────────────────────────
        window = np.hanning(len(self.audio_data))
        yf = fft(self.audio_data * window)
        mag_lin = 2 / self.CHUNK * np.abs(yf[:self.CHUNK // 2])

        # ── Convert to dB ───────────────────────────────────────────
        mag_db = 20 * np.log10(mag_lin + 1e-8)

        # ── Update FFT line ─────────────────────────────────────────
        self.line_fft.set_ydata(mag_db)

        # ── Frequency axis setup ────────────────────────────────────
        freqs = np.fft.rfftfreq(self.CHUNK, 1 / self.RATE)
        self.line_fft.set_xdata(freqs)

        if self.log_freq_checkbox.isChecked():
            self.ax_fft.set_xscale("log")
            self.ax_fft.set_xlim(20, self.RATE / 2)  # Avoid log(0)
        else:
            self.ax_fft.set_xscale("linear")
            self.ax_fft.set_xlim(0, self.RATE / 2)

        # ── Adjust y-axis dB range using slider ─────────────────────
        zoom_db_range = self.zoom_slider.value()  # e.g. 60
        self.ax_fft.set_ylim(-zoom_db_range, 0)

        # ── Redraw canvas ───────────────────────────────────────────
        self.canvas.draw()

    def closeEvent(self, event):
        """Handle window close event"""
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        event.accept()