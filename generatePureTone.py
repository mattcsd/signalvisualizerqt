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

import os

class PureTone(QDialog):

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.selectedAudio = np.empty(1)
        self.default_values = {
            'duration': 1.0,
            'amplitude': 0.5,
            'fs': 44100,
            'offset': 0.0,
            'frequency': 440,
            'phase': 0.0
        }
        self.sliders = {}
        
        self.setupUI()
        self.plotPureTone()
        self.setupAudioInteractions()  # Add this line to initialize audio interactions
        self.help = Help(self)  # Create help system

    def showHelp(self):
        print("Help button clicked")  # Debug
        if hasattr(self, 'help') and self.help:
            print("Help system available")  # Debug
            self.help.createHelpMenu(1)

        else:
            print("Help system not available")  # Debug

        if hasattr(self, 'help') and self.help:
            self.help.createHelpMenu(1)  # 1 corresponds to Pure Tone help
        else:
            # Fallback in case help system isn't initialized
            QMessageBox.information(self, "Help", 
                                   "Pure Tone Generator Help\n\n"
                                   "This tool generates a pure sine wave with adjustable parameters:\n"
                                   "- Duration: Length of the tone in seconds\n"
                                   "- Amplitude: Volume of the tone (0-1)\n"
                                   "- Frequency: Pitch of the tone in Hz\n"
                                   "- Phase: Starting point in the wave cycle\n"
                                   "- Offset: DC offset of the signal")

    def setupUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Math display with fallback font
        self.math_display = QLabel()
        self.math_display.setAlignment(Qt.AlignCenter)
        self.math_display.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-family: "Times New Roman", serif;  /* Fallback for Cambria */
                color: #0066cc;
                background-color: #f0f8ff;
                border: 1px solid #c0c0c0;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 15px;
            }
        """)
        main_layout.addWidget(self.math_display)
        
        # Figure setup
        self.fig = plt.figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(self.create_controls())
        
        self.setLayout(main_layout)
        self.update_expression()

    def setupAudioInteractions(self):
        """Setup audio playback interactions"""
        # Region selection
        self.span = SpanSelector(
            self.ax,
            self.on_select_region,
            'horizontal',
            useblit=True,
            interactive=True,
            drag_from_anywhere=True
        )
        
        # Ensure the span selector stays active
        self.span.set_active(True)

    def on_select_region(self, xmin, xmax):
        """Play selected region of the tone"""
        if len(self.selectedAudio) <= 1:  # No audio generated yet
            return
            
        fs = self.default_values['fs']
        duration = self.sliders['Duration (s)'].value() / 100
        time = np.linspace(0, duration, len(self.selectedAudio), endpoint=False)
        
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
        
        # Create and store sliders
        self.sliders['Duration (s)'] = self.create_slider(0.01, 30.0, self.default_values['duration'])
        self.sliders['Offset'] = self.create_slider(-1.0, 1.0, self.default_values['offset'])
        self.sliders['Amplitude'] = self.create_slider(0.0, 1.0, self.default_values['amplitude'])
        self.sliders['Frequency (Hz)'] = self.create_slider(0, 20000, self.default_values['frequency'], is_float=False)
        self.sliders['Phase (π rad)'] = self.create_slider(-1.0, 1.0, self.default_values['phase'])
        
        # Add to layout
        for i, (label, slider) in enumerate(self.sliders.items()):
            layout.addWidget(QLabel(label), i, 0, alignment=Qt.AlignRight)
            layout.addWidget(slider, i, 1, 1, 2)
            layout.addWidget(self.create_value_display(slider, label.endswith('Hz)')), i, 3)
        
        # Buttons layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(QPushButton('Save', clicked=self.saveDefaults))
        btn_layout.addWidget(QPushButton('Plot', clicked=self.plotPureTone))
        btn_layout.addStretch(1)
        btn_layout.addWidget(QPushButton('Default Values', clicked=self.reset_to_defaults))
        btn_layout.addWidget(QPushButton('🛈 Help', clicked=self.showHelp))
        
        layout.addLayout(btn_layout, len(self.sliders), 1, 1, 3)
        
        return layout

    def update_expression(self):
        """Update the mathematical expression display"""
        if not hasattr(self, 'sliders') or not self.sliders:
            return  # Safety check
            
        pi = "π"
        expression = (
            f"<div style='text-align: center;'>"
            f"<span style='font-size: 22px;'>y(t) = {self.sliders['Offset'].value()/100:.2f} + </span>"
            f"<span style='font-size: 24px; color: #d63333;'>{self.sliders['Amplitude'].value()/100:.2f}</span>"
            f"<span style='font-size: 22px;'>·cos( 2{pi}·</span>"
            f"<span style='font-size: 24px; color: #338033;'>{self.sliders['Frequency (Hz)'].value()}</span>"
            f"<span style='font-size: 22px;'>·t + </span>"
            f"<span style='font-size: 24px; color: #9933cc;'>{self.sliders['Phase (π rad)'].value()/100:.2f}{pi} )</span>"
            f"</div>"
        )
        self.math_display.setText(expression)


    def plotPureTone(self):
        """Generate and plot the pure tone"""
        self.ax.clear()
        
        # Get parameters
        duration = self.sliders['Duration (s)'].value() / 100
        amplitude = self.sliders['Amplitude'].value() / 100
        frequency = self.sliders['Frequency (Hz)'].value()
        phase = self.sliders['Phase (π rad)'].value() / 100
        offset = self.sliders['Offset'].value() / 100
        fs = self.default_values['fs']
        
        # Generate signal
        samples = int(duration * fs)
        time = np.linspace(0, duration, samples, endpoint=False)
        self.selectedAudio = amplitude * np.cos(2*np.pi*frequency*time + phase*np.pi) + offset
        
        # Plot
        self.ax.plot(time, self.selectedAudio, linewidth=1.5, color='blue')
        self.ax.set(xlim=[0, duration], 
                   ylim=[-1.1, 1.1],  # Fixed y-limits for audio signals
                   xlabel='Time (s)', 
                   ylabel='Amplitude')
        self.ax.grid(True, linestyle=':', alpha=0.5)
        
        # Redraw canvas and update expression
        self.canvas.draw()
        self.update_expression()
        
        # Reinitialize span selector after new plot
        self.setupAudioInteractions()






    ''' Audio playing functions '''

    def reset_to_defaults(self):
        """Reset all controls to default values"""
        # Reset sliders
        self.sliders['Duration (s)'].setValue(int(self.default_values['duration'] * 100))
        self.sliders['Offset'].setValue(int(self.default_values['offset'] * 100))
        self.sliders['Amplitude'].setValue(int(self.default_values['amplitude'] * 100))
        self.sliders['Frequency (Hz)'].setValue(self.default_values['frequency'])
        self.sliders['Phase (π rad)'].setValue(int(self.default_values['phase'] * 100))
        
        # Update the display values
        for label, slider in self.sliders.items():
            display = slider.property('display_widget')
            if display:
                if label.endswith('Hz)'):
                    display.setText(str(slider.value()))
                else:
                    display.setText(f"{slider.value()/100:.2f}")

        self.update_expression()  # Update the math display after reset
        
        # Update the plot
        self.plotPureTone()

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


    def saveDefaults(self):
        # Implement your save functionality here
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