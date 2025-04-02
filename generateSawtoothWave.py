import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import unicodedata
from PyQt5.QtWidgets import (QApplication, QDialog, QLabel, QLineEdit, QPushButton, 
                            QRadioButton, QCheckBox, QComboBox, QGridLayout, 
                            QSlider, QMessageBox, QVBoxLayout, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector, Button, RadioButtons

from auxiliar import Auxiliar
from controlMenu import ControlMenu
from help import Help
from scipy import signal

class SawtoothWave(QDialog):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.selectedAudio = np.empty(1)
        self.default_values = {
            'duration': 1.0,
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
        self.help = Help(self)

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
        
        # Math display
        
        
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
        
        # Create sliders
        self.sliders['Duration (s)'] = self.create_slider(0.01, 30.0, self.default_values['duration'])
        self.sliders['Offset'] = self.create_slider(-1.0, 1.0, self.default_values['offset'])
        self.sliders['Amplitude'] = self.create_slider(0.0, 1.0, self.default_values['amplitude'])
        self.sliders['Frequency (Hz)'] = self.create_slider(0, 20000, self.default_values['frequency'], is_float=False)
        self.sliders['Phase (π rad)'] = self.create_slider(-1.0, 1.0, self.default_values['phase'])
        self.sliders['Max Position'] = self.create_slider(0.0, 1.0, self.default_values['maxpos'])
        
        # Add to layout
        for i, (label, slider) in enumerate(self.sliders.items()):
            layout.addWidget(QLabel(label), i, 0, alignment=Qt.AlignRight)
            layout.addWidget(slider, i, 1, 1, 2)
            layout.addWidget(self.create_value_display(slider, label.endswith('Hz)')), i, 3)
        
        # Buttons layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(QPushButton('Save', clicked=self.saveDefaults))
        btn_layout.addWidget(QPushButton('Plot', clicked=self.plotSawtoothWave))
        btn_layout.addStretch(1)
        btn_layout.addWidget(QPushButton('Default Values', clicked=self.reset_to_defaults))
        btn_layout.addWidget(QPushButton('🛈 Help', clicked=self.showHelp))
        
        layout.addLayout(btn_layout, len(self.sliders), 1, 1, 3)
        
        return layout


    def plotSawtoothWave(self):
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
