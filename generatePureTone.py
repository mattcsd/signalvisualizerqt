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

class PureTone(QDialog):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.selectedAudio = np.empty(1)
        
        # Default values
        self.default_values = {
            'duration': 1.0,
            'amplitude': 0.5,
            'fs': 44100,
            'offset': 0.0,
            'frequency': 440,
            'phase': 0.0
        }
        
        # Initialize ControlMenu reference (will be created properly when needed)
        self.cm = None
        
        self.setupUI()
        self.plotPureTone()
        self.setupAudioInteractions()  # Add this line

    def setupUI(self):
        self.setWindowTitle('Generate Pure Tone')
        self.resize(900, 600)
        
        # Create main figure
        self.fig = plt.figure(figsize=(8, 4), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95)
        
        # Create canvas and toolbar
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add figure and controls
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(self.toolbar)
        main_layout.addLayout(self.create_controls())
        
        self.setLayout(main_layout)

    '''* Audio playing functions *'''

    def setupAudioInteractions(self):
        """Setup click-to-play and region selection functionality"""
        # Click-to-play full tone
        # self.canvas.mpl_connect('button_press_event', self.on_click_play)
        
        # Region selection
        self.span = SpanSelector(
            self.ax,
            self.on_select_region,
            'horizontal',
            useblit=True,
            interactive=True,
            drag_from_anywhere=True
        )

    def on_select_region(self, xmin, xmax):
        """Play selected region of the tone"""
        fs = self.default_values['fs']
        time = np.linspace(0, self.sliders['Duration (s)'].value()/100, 
                          len(self.selectedAudio), endpoint=False)
        
        # Find indices for selected region
        idx_min = np.argmax(time >= xmin)
        idx_max = np.argmax(time >= xmax)
        
        # Play the selected portion
        sd.stop()
        sd.play(self.selectedAudio[idx_min:idx_max], fs)

    def create_controls(self):
        layout = QGridLayout()
        layout.setVerticalSpacing(8)
        layout.setHorizontalSpacing(10)
        
        # Create sliders
        self.sliders = {
            'Duration (s)': self.create_slider(0.01, 30.0, self.default_values['duration']),
            'Offset': self.create_slider(-1.0, 1.0, self.default_values['offset']),
            'Amplitude': self.create_slider(0.0, 1.0, self.default_values['amplitude']),
            'Frequency (Hz)': self.create_slider(0, 20000, self.default_values['frequency'], is_float=False),
            'Phase (π rad)': self.create_slider(-1.0, 1.0, self.default_values['phase'])
        }
        
        # Add to layout
        for i, (label, slider) in enumerate(self.sliders.items()):
            layout.addWidget(QLabel(label), i, 0, alignment=Qt.AlignRight)
            layout.addWidget(slider, i, 1, 1, 2)
            layout.addWidget(self.create_value_display(slider, label.endswith('Hz)')), i, 3)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(QPushButton('Save', clicked=self.saveDefaults))
        btn_layout.addWidget(QPushButton('Plot', clicked=self.plotPureTone))
        btn_layout.addWidget(QPushButton('Help', clicked=self.showHelp))
        
        layout.addLayout(btn_layout, len(self.sliders), 1, 1, 3)
        
        return layout

    def create_slider(self, min_val, max_val, init_val, is_float=True):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(min_val*100), int(max_val*100)) if is_float else slider.setRange(min_val, max_val)
        slider.setValue(int(init_val*100)) if is_float else slider.setValue(init_val)
        slider.valueChanged.connect(self.update_plot)
        return slider

    def create_value_display(self, slider, is_int=False):
        value = slider.value() / 100 if not is_int else slider.value()
        label = QLabel(f"{value:.2f}" if not is_int else f"{value}")
        slider.valueChanged.connect(lambda v: label.setText(f"{v/100:.2f}" if not is_int else f"{v}"))
        return label

    def update_plot(self):
        self.plotPureTone()

    def plotPureTone(self):
        self.ax.clear()
        
        # Get parameters
        duration = self.sliders['Duration (s)'].value() / 100
        amplitude = self.sliders['Amplitude'].value() / 100
        frequency = self.sliders['Frequency (Hz)'].value()
        phase = self.sliders['Phase (π rad)'].value() / 100
        offset = self.sliders['Offset'].value() / 100
        fs = self.default_values['fs']
        
        # Generate and store signal
        samples = int(duration * fs)
        time = np.linspace(0, duration, samples, endpoint=False)
        self.selectedAudio = amplitude * np.cos(2*np.pi*frequency*time + phase*np.pi) + offset
        
        # Plot
        self.ax.plot(time, self.selectedAudio, linewidth=1.5, color='blue')

        # Set axes limits
        y_margin = max(0.1, amplitude * 0.2)
        self.ax.set_ylim(-amplitude-y_margin, amplitude+y_margin)
        self.ax.set_xlim(0, duration)
        
        # Add reference lines
        self.ax.axhline(0, color='black', linestyle=':', alpha=0.5)
        self.ax.axhline(1.0, color='red', linestyle='--', alpha=0.3)
        self.ax.axhline(-1.0, color='red', linestyle='--', alpha=0.3)
        self.ax.axhline(offset, color='green', linestyle='-.', alpha=0.5)
        
        # Configure grid and labels
        self.ax.grid(True, linestyle=':', alpha=0.5)
        self.ax.set_xlabel('Time (s)', fontsize=9)
        self.ax.set_ylabel('Amplitude', fontsize=9)
        
        # Redraw canvas
        self.canvas.draw()

    def saveDefaults(self):
        # Implement your save functionality here
        pass

    def showHelp(self):
        # Implement help functionality
        pass

    def createControlMenu(self):
        """Create the ControlMenu when needed with proper parameters"""
        duration = self.sliders['Duration (s)'].value() / 100
        fs = self.default_values['fs']
        signal = self.selectedAudio
        name = "Pure Tone"
        
        # Create ControlMenu with required parameters
        self.cm = ControlMenu(name, fs, signal, duration, self.controller)
        self.cm.show()