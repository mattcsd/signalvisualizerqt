import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import unicodedata
from PyQt5.QtWidgets import (QApplication, QWidget, QDialog, QLabel, QLineEdit, QPushButton, 
                            QRadioButton, QCheckBox, QComboBox, QGridLayout, 
                            QSlider, QMessageBox, QVBoxLayout, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector, Button, RadioButtons

from auxiliar import Auxiliar
from controlMenu import ControlMenu
from scipy import signal

class SawtoothWave(QWidget):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.master = master
        self.selectedAudio = np.empty(1)
        self.default_values = {
            'duration': 0.3,
            'amplitude': 0.8,
            'fs': 44100,
            'offset': 0.0,
            'frequency': 440,
            'phase': 0.0,
            'maxpos': 1.0
        }
        self.sliders = {}

        self.setupUI()
        self.plotSawtoothWave()
        self.setupAudioInteractions()

    def showHelp(self):
        if hasattr(self, 'help') and self.help:
            self.help.openHelpPage('sawtooth_help.html')  # 2 corresponds to Sawtooth help
        else:
            QMessageBox.information(self, "Help", 
                                   "Sawtooth Wave Generator Help\n\n"
                                   "This tool generates a sawtooth wave with adjustable parameters:\n"
                                   "- Duration: Length in seconds\n"
                                   "- Amplitude: Volume (0-1)\n"
                                   "- Frequency: Pitch in Hz\n"
                                   "- Phase: Starting point in cycle\n"
                                   "- Max Position: Width of the ramp (0-1)\n"
                                   "- Offset: DC offset")

    def setupUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        

        
        # Figure setup
        self.fig = plt.figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(self.create_controls())
        
        self.setLayout(main_layout)

    def setupAudioInteractions(self):
        self.span = SpanSelector(
            self.ax,
            self.on_select_region,
            'horizontal',
            useblit=True,
            interactive=True,
            drag_from_anywhere=True
        )
        self.span.set_active(True)

    def on_select_region(self, xmin, xmax):
        if len(self.selectedAudio) <= 1:
            return
            
        fs = self.default_values['fs']
        duration = self.sliders['Duration (s)'].value() / 100
        time = np.linspace(0, duration, len(self.selectedAudio), endpoint=False)
        
        idx_min = np.argmax(time >= xmin)
        idx_max = np.argmax(time >= xmax)
        
        sd.stop()
        sd.play(self.selectedAudio[idx_min:idx_max], fs)

    def create_controls(self):
        layout = QGridLayout()
        layout.setVerticalSpacing(8)
        layout.setHorizontalSpacing(10)
        
        self.sliders['Duration (s)'] = self.create_slider(0.01, 30.0, self.default_values['duration'])
        self.sliders['Offset'] = self.create_slider(-1.0, 1.0, self.default_values['offset'])
        self.sliders['Amplitude'] = self.create_slider(0.0, 1.0, self.default_values['amplitude'])
        self.sliders['Frequency (Hz)'] = self.create_slider(0, 20000, self.default_values['frequency'], is_float=False)
        self.sliders['Phase (π rad)'] = self.create_slider(-1.0, 1.0, self.default_values['phase'])
        self.sliders['Max Position'] = self.create_slider(0.0, 1.0, self.default_values['maxpos'])
        
        for i, (label, slider) in enumerate(self.sliders.items()):
            layout.addWidget(QLabel(label), i, 0, alignment=Qt.AlignRight)
            layout.addWidget(slider, i, 1, 1, 2)
            layout.addWidget(self.create_value_display(slider, label.endswith('Hz)')), i, 3)
        
        btn_layout = QHBoxLayout()
        
        self.save_button = QPushButton('Save')
        self.help_button = QPushButton('🛈')
        self.controller_button = QPushButton('Load to Controller')
        self.plot_button = QPushButton('Plot')
        
        self.help_button.setFixedWidth(30)
        
        btn_layout.addWidget(self.save_button)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.controller_button)
        btn_layout.addWidget(self.help_button)
        btn_layout.addWidget(self.plot_button)

        
        self.help_button.clicked.connect(lambda: self.controller.help.createHelpMenu(4))
        self.plot_button.clicked.connect(self.plotSawtoothWave)
        self.controller_button.clicked.connect(self.load_to_controller)
        self.save_button.clicked.connect(self.saveDefaults)
        
        layout.addLayout(btn_layout, len(self.sliders), 1, 1, 3)
        
        return layout

    def plotSawtoothWave(self):
        # Clear any existing span selector first
        if hasattr(self, 'span'):
            self.span.clear()
            del self.span
            
        self.ax.clear()
        
        # Get parameters
        duration = self.sliders['Duration (s)'].value() / 100
        amplitude = self.sliders['Amplitude'].value() / 100
        frequency = self.sliders['Frequency (Hz)'].value()
        phase = self.sliders['Phase (π rad)'].value() / 100
        offset = self.sliders['Offset'].value() / 100
        maxpos = self.sliders['Max Position'].value() / 100
        fs = self.default_values['fs']
        
        # Generate signal
        samples = int(duration * fs)
        time = np.linspace(0, duration, samples, endpoint=False)
        self.selectedAudio = amplitude * signal.sawtooth(2*np.pi*frequency*time + phase*np.pi, width=maxpos) + offset
        
        # Plot
        self.ax.plot(time, self.selectedAudio, linewidth=1.5, color='blue')
        self.ax.set(xlim=[0, duration], 
                   ylim=[-1.1, 1.1],
                   xlabel='Time (s)', 
                   ylabel='Amplitude')
        self.ax.grid(True, linestyle=':', alpha=0.5)
        
        self.canvas.draw()
        self.setupAudioInteractions()

    def reset_to_defaults(self):
        for name, value in self.default_values.items():
            if name == 'frequency':
                self.sliders['Frequency (Hz)'].setValue(value)
            elif name == 'fs':
                continue  # Not adjustable via slider
            else:
                slider_name = {
                    'duration': 'Duration (s)',
                    'amplitude': 'Amplitude',
                    'offset': 'Offset',
                    'phase': 'Phase (π rad)',
                    'maxpos': 'Max Position'
                }.get(name)
                if slider_name:
                    self.sliders[slider_name].setValue(int(value * 100))
        
        self.plotSawtoothWave()

    def create_slider(self, min_val, max_val, init_val, is_float=True):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(min_val*100), int(max_val*100)) if is_float else slider.setRange(min_val, max_val)
        slider.setValue(int(init_val*100)) if is_float else slider.setValue(init_val)
        slider.valueChanged.connect(self.update_plot)
        return slider

    def create_value_display(self, slider, is_int=False):
        value = slider.value() / 100 if not is_int else slider.value()
        input_field = QLineEdit(f"{value:.2f}" if not is_int else f"{value}")
        input_field.setFixedWidth(50)
        input_field.setAlignment(Qt.AlignCenter)
        input_field.returnPressed.connect(lambda: self.update_slider_from_input(slider, input_field, is_int))
        slider.valueChanged.connect(lambda v: input_field.setText(f"{v/100:.2f}" if not is_int else f"{v}"))
        return input_field

    def load_to_controller(self):
        """Load the generated sawtooth wave to a new controller window."""
        try:
            # Ensure the waveform is freshly generated with current slider values
            self.plotSawtoothWave()

            # Use selected span if valid, otherwise full audio
            audio_to_load = (
                self.selectedAudio if self.selectedAudio.size > 1 else self.selectedAudio
            )
            duration = self.sliders['Duration (s)'].value() / 100
            fs = self.default_values['fs']

            # Compose a descriptive title
            frequency = self.sliders['Frequency (Hz)'].value()
            maxpos = self.sliders['Max Position'].value() / 100
            title = f"Sawtooth Wave {frequency}Hz (MaxPos: {maxpos:.2f})"

            # Check if controller is defined
            if not hasattr(self.controller, 'adse'):
                from PyQt5.QtWidgets import QWidget
                self.controller = QWidget()
                self.controller.adse = type('', (), {})()
                self.controller.adse.advancedSettings = lambda: print("Advanced settings not available")

            # Create and show the controller window
            control_window = ControlMenu(title, fs, audio_to_load, duration, self.controller)

            # Optional: Track and clean up the controller window
            if not hasattr(self, 'control_windows'):
                self.control_windows = []
            self.control_windows.append(control_window)

            control_window.destroyed.connect(
                lambda: self.control_windows.remove(control_window)
                if control_window in self.control_windows else None
            )

            control_window.show()
            control_window.activateWindow()

        except Exception as e:
            print(f"Error loading sawtooth wave to controller: {e}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Could not load to controller:\n{str(e)}")

    def update_plot(self):
        self.plotSawtoothWave()

    def saveDefaults(self):
        # Implement your save functionality here
        pass

    def createControlMenu(self):
        duration = self.sliders['Duration (s)'].value() / 100
        fs = self.default_values['fs']
        signal = self.selectedAudio
        name = "Sawtooth Wave"
        
        self.cm = ControlMenu(name, fs, signal, duration, self.controller)
        self.cm.show()

    def update_slider_from_input(self, slider, input_field, is_int):
        try:
            value = float(input_field.text()) if not is_int else int(input_field.text())
            slider.setValue(int(value * 100) if not is_int else value)
        except ValueError:
            input_field.setText(f"{slider.value()/100:.2f}" if not is_int else f"{slider.value()}")
