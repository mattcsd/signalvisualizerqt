import numpy as np
import pyaudio
from scipy.fft import fft
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QPushButton, QCheckBox, QWidget, QSlider, QVBoxLayout, QComboBox, QLabel, 
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
        self.zoom_level = 60  # Initial zoom level (dB range)
        self.min_zoom = 20    # Minimum zoom level (more zoomed in)
        self.max_zoom = 120   # Maximum zoom level (more zoomed out)
        
        # Instrument frequencies (in Hz)
        self.instrument_frequencies = {
            'Guitar (Standard)': [82.41, 110.00, 146.83, 196.00, 246.94, 329.63],  # E2, A2, D3, G3, B3, E4
            'Violin': [196.00, 293.66, 440.00, 659.26],  # G3, D4, A4, E5
            'Cretan Lute': [82.41, 110.00, 146.83, 196.00],  # E A D G
            'Piano': [27.50, 55.00, 110.00, 220.00, 440.00, 880.00]  # A0-A5
        }
        
        self.instrument_labels = {
            'Guitar (Standard)': ['E2', 'A2', 'D3', 'G3', 'B3', 'E4'],
            'Violin': ['G3', 'D4', 'A4', 'E5'],
            'Cretan Lute': ['E', 'A', 'D', 'G'],
            'Piano': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
        }
        
        self.freq_markers = []  # To store reference line objects
        self.freq_labels = []   # To store text label objects

        # Setup matplotlib figure and canvas
        self.setup_ui()

        # Start audio stream
        self.start_audio_stream()
        
        # Data buffers
        self.audio_data = np.zeros(self.CHUNK)
        self.fft_data = np.zeros(self.CHUNK//2)
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

        # Instrument dropdown setup
        self.instrument_label = QLabel("Instrument:")
        self.instrument_dropdown = QComboBox()
        self.instrument_dropdown.addItem("-- Select Instrument --", None)  # Default empty option
        self.instrument_dropdown.addItems(self.instrument_frequencies.keys())
        self.instrument_dropdown.currentTextChanged.connect(self.update_instrument_markers)


        # Add reset button
        self.reset_button = QPushButton("Reset Values")
        self.reset_button.clicked.connect(self.reset_values)
        
        # Device selection dropdown
        self.device_label = QLabel("Input Device:")
        self.device_dropdown = QComboBox()
        self.populate_device_dropdown()
        self.device_dropdown.currentIndexChanged.connect(self.change_device)
        
        # Gain slider
        self.zoom_label = QLabel("Gain:")
        self.zoom_slider = QSlider()
        self.zoom_slider.setOrientation(Qt.Horizontal)
        self.zoom_slider.setRange(10, 120)  # 10=very amplified, 120=very attenuated
        self.zoom_slider.setValue(60)       # Default neutral position
        self.zoom_slider.valueChanged.connect(self.update_zoom_level)

        # Offset slider
        self.offset_label = QLabel("Vertical Offset:")
        self.offset_slider = QSlider()
        self.offset_slider.setOrientation(Qt.Horizontal)
        self.offset_slider.setRange(-40, 40)  # -40dB to +40dB offset range
        self.offset_slider.setValue(0)        # Default no offset
        self.offset_slider.valueChanged.connect(self.update_plot)


        # Log/linear frequency scale checkbox
        self.log_freq_checkbox = QCheckBox("Linear Frequency")
        self.log_freq_checkbox.setChecked(False)
        self.log_freq_checkbox.stateChanged.connect(self.update_plot)
        
        # Add widgets to control panel
        control_layout.addWidget(self.device_label)
        control_layout.addWidget(self.device_dropdown)
        control_layout.addWidget(self.zoom_label)
        control_layout.addWidget(self.zoom_slider)
        control_layout.addWidget(self.offset_label)
        control_layout.addWidget(self.offset_slider)
        control_layout.addWidget(self.log_freq_checkbox)

        control_layout.addWidget(self.instrument_label)
        control_layout.addWidget(self.instrument_dropdown)
        control_layout.addWidget(self.reset_button)
        
        # Add stretch to push controls left
        control_layout.addStretch()
        
        # Add widgets to main layout
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(control_panel)
        
        # Setup plots
        self.setup_plots()

    def reset_values(self):
        """Reset all controls to default values"""
        self.zoom_slider.setValue(60)
        self.offset_slider.setValue(0)
        self.log_freq_checkbox.setChecked(False)
        self.update_plot()

    def update_instrument_markers(self, instrument_name):
        """Update the frequency reference lines based on selected instrument"""
        # Clear existing markers and labels
        for marker in self.freq_markers:
            marker.remove()
        for label in self.freq_labels:
            label.remove()
        self.freq_markers.clear()
        self.freq_labels.clear()

        # Skip if no instrument selected or it's the placeholder
        if not instrument_name or instrument_name == "-- Select Instrument --":
            self.canvas.draw()
            return

        # Get frequencies and labels for selected instrument
        frequencies = self.instrument_frequencies.get(instrument_name, [])
        labels = self.instrument_labels.get(instrument_name, [])

        # Create vertical lines and labels for each frequency
        colors = [
                '#FF5733',  # Red-orange
                '#33FF57',  # Green
                '#3357FF',  # Blue
                '#F033FF',  # Purple 
                '#FFC733',  # Yellow-orange (new)
                '#33FFF0'   # Cyan (new)
        ]
        
        # Calculate dynamic vertical positions (higher on the plot)
        num_notes = len(frequencies)
        base_y = 25  # Starting y position (higher up)
        y_step = 20   # Vertical spacing between labels
        y_positions = [base_y - (i % 3) * y_step for i in range(num_notes)]
        angle = 30
        
        for freq, label, color, y_pos in zip(frequencies, labels, colors, y_positions):
            # Add vertical line
            marker = self.ax_fft.axvline(x=freq, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            self.freq_markers.append(marker)
            
            # Add text label without box
            text = self.ax_fft.text(
                freq, y_pos, f"{label} ({freq:.1f}Hz)", 
                color=color, 
                ha='center', 
                va='top',
                fontsize=10,
                alpha=0.9,
                weight='bold',
                rotation=angle,
                rotation_mode='anchor'
            )
            self.freq_labels.append(text)

        # Adjust ylim to ensure labels are visible
        current_ylim = self.ax_fft.get_ylim()
        self.ax_fft.set_ylim(current_ylim[0], current_ylim[1])  # Keep existing upper limit
        
        self.canvas.draw()

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
        self.ax_wave.set_ylabel('Amplitude')
        self.ax_wave.grid(True)
        
        # FFT plot initialization
        freqs = np.fft.rfftfreq(self.CHUNK, 1 / self.RATE)
        self.line_fft, = self.ax_fft.semilogx(freqs, np.zeros_like(freqs), 'r')
        
        self.setup_log_ticks()

        self.ax_fft.set_title('Frequency Domain - FFT Analysis')
        self.ax_fft.set_xlim(*self.frequency_range)
        self.ax_fft.set_ylim(-self.zoom_level, 0)
        self.ax_fft.set_xlabel('Frequency (Hz)')
        self.ax_fft.set_ylabel('Magnitude (dB)')
        self.ax_fft.grid(True, which='both')

    def update_zoom_level(self, value):
        """Update the zoom level (dB range) for the FFT plot"""
        self.zoom_level = value
        self.ax_fft.set_ylim(-self.zoom_level, 0)
        self.canvas.draw()

    def change_device(self, index):
        """Handle device selection change"""
        device_index = self.device_dropdown.itemData(index)
        print(f"Selected device index: {device_index}")
        self.start_audio_stream(device_index)

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

    def setup_log_ticks(self):
        """Helper function to configure log scale ticks"""
        # Set manual tick positions at specific frequencies
        tick_positions = []
        tick_labels = []
        
        # Define our frequency steps
        steps_under_100 = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        steps_100_to_1000 = [
            100, 125, 150, 175, 200, 250, 300, 350, 
            400, 500, 600, 700, 800, 900, 1000
        ]
        steps_above_1000 = [1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        
        # Combine all steps
        all_steps = steps_under_100 + steps_100_to_1000 + steps_above_1000
        
        # Create ticks and labels
        for freq in all_steps:
            if 20 <= freq <= 20000:  # Within our display range
                tick_positions.append(freq)
                if freq < 1000:
                    tick_labels.append(f"{freq:.0f}")
                else:
                    if freq % 1000 == 0:
                        tick_labels.append(f"{freq//1000}k")
                    else:
                        tick_labels.append(f"{freq/1000:.1f}k")
        
        # Set the ticks
        self.ax_fft.set_xticks(tick_positions)
        self.ax_fft.set_xticklabels(tick_labels, rotation=45, ha='right', rotation_mode='anchor')        
        
        # Configure minor ticks (more dense)
        self.ax_fft.xaxis.set_minor_locator(plt.LogLocator(
            base=10, 
            subs=np.linspace(0.1, 1.0, 10)[1:-1]  # More minor ticks between majors
        ))
        self.ax_fft.minorticks_on()

        # Adjust layout to prevent label cutoff
        self.figure.tight_layout()
        self.figure.subplots_adjust(bottom=0.15)  # Add more space at bottom
    
        
        # Style adjustments
        self.ax_fft.tick_params(axis='x', which='major', length=6, width=1)
        self.ax_fft.tick_params(axis='x', which='minor', length=3, width=0.5)

        #self.ax_fft.minorticks_on()

    def update_plot(self):
        """Update the plots with new audio data"""
        if not self.running or not hasattr(self, 'audio_data'):
            return

        # Apply zoom/gain to the raw audio data
        zoom_factor = self.zoom_level / 60.0
        processed_audio = self.audio_data * zoom_factor

        # Update waveform plot
        self.line_wave.set_ydata(processed_audio)
        self.ax_wave.set_ylim(-1, 1)

        # Compute FFT with Hann window
        window = np.hanning(len(processed_audio))
        yf = fft(processed_audio * window)
        mag_lin = 2 / self.CHUNK * np.abs(yf[:len(yf)//2 + 1])
        mag_db = 20 * np.log10(mag_lin + 1e-8) + self.offset_slider.value()

        # Update FFT line data
        freqs = np.fft.rfftfreq(self.CHUNK, 1 / self.RATE)
        min_length = min(len(freqs), len(mag_db))
        self.line_fft.set_data(freqs[:min_length], mag_db[:min_length])

        # Handle scale type
        if self.log_freq_checkbox.isChecked():
            self.ax_fft.set_xscale("linear")
            self.ax_fft.set_xlim(0, self.RATE / 2)
            self.ax_fft.xaxis.set_major_formatter(plt.ScalarFormatter())
            self.ax_fft.xaxis.set_major_locator(plt.MaxNLocator(10))
        else:
            self.ax_fft.set_xscale("log")
            self.setup_log_ticks()  # Reapply our custom log ticks
                    
        # Remove any zero lines (including red line)
        for line in self.ax_fft.lines:
            if len(line.get_ydata()) > 0 and np.all(line.get_ydata() == 0):
                line.remove()

        # Adjust y-limits
        max_offset = self.offset_slider.maximum()
        self.ax_fft.set_ylim(-60 - max_offset, 0 + max_offset)

        self.canvas.draw()

    def update_zoom_level(self, value):
        """Update the zoom/gain level (1.0 = normal)"""
        self.zoom_level = value
        self.update_plot()  # Trigger full update to see changes

    def closeEvent(self, event):
        """Handle window close event"""
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        event.accept()



