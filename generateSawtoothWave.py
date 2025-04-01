import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd
import unicodedata
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.widgets import SpanSelector, Button, RadioButtons
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QSlider, QLineEdit, QPushButton, QDialog, QMessageBox)
from PyQt5.QtCore import Qt

from auxiliar import Auxiliar
from controlMenu import ControlMenu
from PyQt5.QtGui import QDoubleValidator, QIntValidator

class SawtoothWave(QWidget):
    def __init__(self, container, controller, parent=None):
        super().__init__(parent)
        self.container = container
        self.controller = controller
        self.aux = Auxiliar()
        self.cm = None
        self.selectedAudio = np.empty(1)
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Generate sawtooth wave')
        self.setWindowIcon(self.controller.icons['icon'])
        
        # Create main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # Add widgets to layout
        main_layout.addWidget(toolbar)
        main_layout.addWidget(self.canvas)
        
        # Create control widgets FIRST
        self.create_controls()
        
        # THEN do the initial plot
        self.plotSawtoothWave()

    def connect_signals(self):
        # Sliders and edits
        self.dur_slider.valueChanged.connect(lambda val: self.slider_changed(val, self.dur_edit, 100))
        self.dur_edit.editingFinished.connect(lambda: self.edit_changed(self.dur_edit, self.dur_slider, 100))
        
        self.off_slider.valueChanged.connect(lambda val: self.slider_changed(val, self.off_edit, 100))
        self.off_edit.editingFinished.connect(lambda: self.edit_changed(self.off_edit, self.off_slider, 100))
        
        self.amp_slider.valueChanged.connect(lambda val: self.slider_changed(val, self.amp_edit, 100))
        self.amp_edit.editingFinished.connect(lambda: self.edit_changed(self.amp_edit, self.amp_slider, 100))
        
        self.freq_slider.valueChanged.connect(lambda val: self.slider_changed(val, self.freq_edit, 1))
        self.freq_edit.editingFinished.connect(lambda: self.edit_changed(self.freq_edit, self.freq_slider, 1))
        
        self.phase_slider.valueChanged.connect(lambda val: self.slider_changed(val, self.phase_edit, 100))
        self.phase_edit.editingFinished.connect(lambda: self.edit_changed(self.phase_edit, self.phase_slider, 100))
        
        self.maxp_slider.valueChanged.connect(lambda val: self.slider_changed(val, self.maxp_edit, 100))
        self.maxp_edit.editingFinished.connect(lambda: self.edit_changed(self.maxp_edit, self.maxp_slider, 100))
        
        # Buttons
        self.plot_button.clicked.connect(self.plotSawtoothWave)
        self.save_button.clicked.connect(self.saveDefaultValues)
        self.help_button.clicked.connect(lambda: self.controller.help.createHelpMenu(4))
        
        # FS edit
        self.fs_edit.editingFinished.connect(self.check_fs)
        
    def create_controls(self):
        # Read default values with error handling
        list = self.aux.readFromCsv()
        
        # Set default values in case the CSV doesn't have them
        defaults = {
            'duration': 1.0,
            'amplitude': 0.8,
            'fs': 44100,
            'offset': 0.0,
            'frequency': 440,
            'phase': 0.0,
            'maxpos': 1.0
        }
        
        try:
            # Try to get values from CSV, fall back to defaults if not available
            duration = float(list[3][2]) if len(list) > 3 and len(list[3]) > 2 else defaults['duration']
            amplitude = float(list[3][4]) if len(list) > 3 and len(list[3]) > 4 else defaults['amplitude']
            self.fs = int(list[3][6]) if len(list) > 3 and len(list[3]) > 6 else defaults['fs']
            offset = float(list[3][8]) if len(list) > 3 and len(list[3]) > 8 else defaults['offset']
            frequency = int(list[3][10]) if len(list) > 3 and len(list[3]) > 10 else defaults['frequency']
            phase = float(list[3][12]) if len(list) > 3 and len(list[3]) > 12 else defaults['phase']
            maxpos = float(list[3][14]) if len(list) > 3 and len(list[3]) > 14 else defaults['maxpos']
        except (IndexError, ValueError):
            # If any error occurs, use all defaults
            duration = defaults['duration']
            amplitude = defaults['amplitude']
            self.fs = defaults['fs']
            offset = defaults['offset']
            frequency = defaults['frequency']
            phase = defaults['phase']
            maxpos = defaults['maxpos']

        # Create control widgets and assign them as instance variables
        controls_layout = QVBoxLayout()
        
        # Duration control
        dur_layout = QHBoxLayout()
        self.dur_label = QLabel('Total duration (s)')
        self.dur_slider = QSlider(Qt.Horizontal)
        self.dur_slider.setRange(1, 3000)  # 0.01 to 30.00 in steps of 0.01
        self.dur_slider.setValue(int(float(duration)*100))
        self.dur_edit = QLineEdit(str(duration))
        self.dur_edit.setFixedWidth(80)
        self.dur_edit.setValidator(QDoubleValidator())
        dur_layout.addWidget(self.dur_label)
        dur_layout.addWidget(self.dur_slider)
        dur_layout.addWidget(self.dur_edit)
        controls_layout.addLayout(dur_layout)
        
        # Offset control
        off_layout = QHBoxLayout()
        self.off_label = QLabel('Offset')
        self.off_slider = QSlider(Qt.Horizontal)
        self.off_slider.setRange(-100, 100)  # -1.00 to 1.00 in steps of 0.01
        self.off_slider.setValue(int(float(offset)*100))
        self.off_edit = QLineEdit(str(offset))
        self.off_edit.setFixedWidth(80)
        self.off_edit.setValidator(QDoubleValidator())
        off_layout.addWidget(self.off_label)
        off_layout.addWidget(self.off_slider)
        off_layout.addWidget(self.off_edit)
        controls_layout.addLayout(off_layout)
        
        # Amplitude control
        amp_layout = QHBoxLayout()
        self.amp_label = QLabel('Amplitude')
        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(0, 100)  # 0.00 to 1.00 in steps of 0.01
        self.amp_slider.setValue(int(float(amplitude)*100))
        self.amp_edit = QLineEdit(str(amplitude))
        self.amp_edit.setFixedWidth(80)
        self.amp_edit.setValidator(QDoubleValidator())
        amp_layout.addWidget(self.amp_label)
        amp_layout.addWidget(self.amp_slider)
        amp_layout.addWidget(self.amp_edit)
        controls_layout.addLayout(amp_layout)
        
        # Frequency control
        freq_layout = QHBoxLayout()
        self.freq_label = QLabel('Frequency (Hz)')
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(0, 24000)  # 0 to 24000 Hz
        self.freq_slider.setValue(int(frequency))
        self.freq_edit = QLineEdit(str(frequency))
        self.freq_edit.setFixedWidth(80)
        self.freq_edit.setValidator(QIntValidator())
        freq_layout.addWidget(self.freq_label)
        freq_layout.addWidget(self.freq_slider)
        freq_layout.addWidget(self.freq_edit)
        controls_layout.addLayout(freq_layout)
        
        # Phase control
        phase_layout = QHBoxLayout()
        self.phase_label = QLabel(f'Phase ({unicodedata.lookup("GREEK SMALL LETTER PI")} rad)')
        self.phase_slider = QSlider(Qt.Horizontal)
        self.phase_slider.setRange(-100, 100)  # -1.00 to 1.00 in steps of 0.01
        self.phase_slider.setValue(int(float(phase)*100))
        self.phase_edit = QLineEdit(str(phase))
        self.phase_edit.setFixedWidth(80)
        self.phase_edit.setValidator(QDoubleValidator())
        phase_layout.addWidget(self.phase_label)
        phase_layout.addWidget(self.phase_slider)
        phase_layout.addWidget(self.phase_edit)
        controls_layout.addLayout(phase_layout)
        
        # Max position control
        maxp_layout = QHBoxLayout()
        self.maxp_label = QLabel('Max. position')
        self.maxp_slider = QSlider(Qt.Horizontal)
        self.maxp_slider.setRange(0, 100)  # 0.00 to 1.00 in steps of 0.01
        self.maxp_slider.setValue(int(float(maxpos)*100))
        self.maxp_edit = QLineEdit(str(maxpos))
        self.maxp_edit.setFixedWidth(80)
        self.maxp_edit.setValidator(QDoubleValidator())
        maxp_layout.addWidget(self.maxp_label)
        maxp_layout.addWidget(self.maxp_slider)
        maxp_layout.addWidget(self.maxp_edit)
        controls_layout.addLayout(maxp_layout)
        
        # Sample rate control
        fs_layout = QHBoxLayout()
        self.fs_label = QLabel('Fs (Hz)')
        self.fs_edit = QLineEdit(str(self.fs))
        self.fs_edit.setFixedWidth(80)
        self.fs_edit.setValidator(QIntValidator())
        fs_layout.addWidget(self.fs_label)
        fs_layout.addWidget(self.fs_edit)
        controls_layout.addLayout(fs_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.plot_button = QPushButton('Plot')
        self.save_button = QPushButton('Save')
        self.help_button = QPushButton('🛈')
        self.help_button.setFixedWidth(30)
        
        button_layout.addWidget(self.plot_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.help_button)
        controls_layout.addLayout(button_layout)
        
        # Add controls to main layout
        self.layout().addLayout(controls_layout)
        
        # Connect signals
        self.connect_signals()
        
    def slider_changed(self, value, edit, divisor):
        edit.setText(str(value / divisor))
        self.plotSawtoothWave()
        
    def edit_changed(self, edit, slider, multiplier):
        try:
            value = float(edit.text())
            slider.setValue(int(value * multiplier))
            self.plotSawtoothWave()
        except ValueError:
            pass
            
    def check_fs(self):
        try:
            fs = int(self.fs_edit.text())
            if fs > 48000:
                self.fs_edit.setText('48000')
                QMessageBox.warning(self, 'Wrong sample frequency value', 
                                  'The sample frequency cannot be greater than 48000 Hz.')
            else:
                self.fs = fs
                self.plotSawtoothWave()
        except ValueError:
            pass
            
    def saveDefaultValues(self):
        list = self.aux.readFromCsv()
        amplitude = float(self.amp_edit.text())
        frequency = int(self.freq_edit.text())
        phase = float(self.phase_edit.text())
        maxpos = float(self.maxp_edit.text())
        duration = float(self.dur_edit.text())
        offset = float(self.off_edit.text())
        fs = int(self.fs_edit.text())

        new_list = [
            ['NOISE','\t duration', list[0][2],'\t amplitude', list[0][4],'\t fs', list[0][6],'\t noise type', list[0][8]],
            ['PURE TONE','\t duration', list[1][2],'\t amplitude', list[1][4],'\t fs', list[1][6],'\t offset', list[1][8],'\t frequency', list[1][10],'\t phase',  list[1][12]],
            ['SQUARE WAVE','\t duration', list[2][2],'\t amplitude', list[2][4],'\t fs', list[2][6],'\t offset', list[2][8],'\t frequency', list[2][10],'\t phase', list[2][12],'\t active cycle', list[2][14]],
            ['SAWTOOTH WAVE','\t duration', duration,'\t amplitude', amplitude,'\t fs', fs,'\t offset', offset,'\t frequency', frequency,'\t phase', phase,'\t max position', maxpos],
            ['FREE ADD OF PT','\t duration', list[4][2],'\t octave', list[4][4],'\t freq1', list[4][6],'\t freq2', list[4][8],'\t freq3', list[4][10],'\t freq4', list[4][12],'\t freq5', list[4][14],'\t freq6', list[4][16],'\t amp1', list[4][18],'\t amp2', list[4][20],'\t amp3', list[4][22],'\t amp4', list[4][24],'\t amp5', list[4][26],'\t amp6', list[4][28]],
            ['SPECTROGRAM','\t colormap', list[5][2]]
        ]
        self.aux.saveDefaultAsCsv(new_list)
        QMessageBox.information(self, 'Saved', 'Default values saved successfully.')
            
    def plotSawtoothWave(self):
        try:
            # Make sure all required attributes exist
            if not hasattr(self, 'amp_edit') or not hasattr(self, 'freq_edit') or \
               not hasattr(self, 'phase_edit') or not hasattr(self, 'maxp_edit') or \
               not hasattr(self, 'dur_edit') or not hasattr(self, 'off_edit'):
                return
                
            amplitude = float(self.amp_edit.text())
            frequency = int(self.freq_edit.text())
            phase = float(self.phase_edit.text())
            maxpos = float(self.maxp_edit.text())
            duration = float(self.dur_edit.text())
            offset = float(self.off_edit.text())
            samples = int(duration * self.fs)
            amplitude = float(self.amp_edit.text())
            frequency = int(self.freq_edit.text())
            phase = float(self.phase_edit.text())
            maxpos = float(self.maxp_edit.text())
            duration = float(self.dur_edit.text())
            offset = float(self.off_edit.text())
            samples = int(duration * self.fs)

            # Check frequency
            self.aux.bigFrequency(frequency, self.fs)

            time = np.linspace(start=0, stop=duration, num=samples, endpoint=False)
            sawtooth = amplitude * signal.sawtooth(2*np.pi*frequency*time + phase*np.pi, width=maxpos) + offset * np.ones(len(time))

            # Clear and redraw plot
            self.ax.clear()
            
            # Plot the sawtooth wave
            limite = max(abs(sawtooth))*1.1
            self.ax.plot(time, sawtooth)
            self.fig.canvas.manager.set_window_title('Sawtooth signal')
            self.ax.set(xlim=[0, duration], ylim=[-limite, limite], 
                       xlabel='Time (s)', ylabel='Amplitude')
            self.ax.axhline(y=0, color='black', linewidth='0.5', linestyle='--')
            self.ax.axhline(y=1.0, color='red', linewidth='0.8', linestyle='--')
            self.ax.axhline(y=-1.0, color='red', linewidth='0.8', linestyle='--')
            self.ax.axhline(y=offset, color='blue', linewidth='1', label="offset")
            self.ax.legend(loc="upper right")
            
            # Add load button and span selector
            self.addLoadButton(self.fig, self.ax, self.fs, time, sawtooth, duration, 'Sawtooth wave')
            self.addScaleSaturateRadiobuttons(self.fig, offset)
            
            self.canvas.draw()
        except Exception as e:
            print(f"Error in plotSawtoothWave: {e}")
        
    def addLoadButton(self, fig, ax, fs, time, audio, duration, name):
        axload = fig.add_axes([0.8, 0.01, 0.09, 0.05])
        but_load = Button(axload, 'Load')
        
        def load(event):
            if self.selectedAudio.shape == (1,): 
                # Initialize ControlMenu with current audio data
                self.cm = ControlMenu(
                    fileName=name,
                    fs=fs,
                    audioFrag=audio,
                    duration=duration,
                    controller=self.controller
                )
                self.cm.createControlMenu()
            else:
                time = np.arange(0, len(self.selectedAudio)/fs, 1/fs)
                durSelec = max(time)
                self.cm = ControlMenu(
                    fileName=name,
                    fs=fs,
                    audioFrag=self.selectedAudio,
                    duration=durSelec,
                    controller=self.controller
                )
                self.cm.createControlMenu()
            self.close()
            
        but_load.on_clicked(load)
        axload._but_load = but_load
        
        def listenFrag(xmin, xmax):
            ini, end = np.searchsorted(time, (xmin, xmax))
            self.selectedAudio = audio[ini:end+1]
            sd.play(self.selectedAudio, fs)
            
        self.span = SpanSelector(ax, listenFrag, 'horizontal', useblit=True, 
                               interactive=True, drag_from_anywhere=True)


    def addScaleSaturateRadiobuttons(self, fig, offset):
        if offset > 0.5 or offset < -0.5:
            rax = fig.add_axes([0.75, 0.9, 0.15, 0.1])
            radio = RadioButtons(rax, ('scale', 'saturate'))
            
            def exceed(label):
                options = {'scale': 0, 'saturate': 1}
                option = options[label]
                if option == 0:
                    for i in range(len(self.selectedAudio)):
                        if self.selectedAudio[i] > 1:
                            self.selectedAudio[i] = 1
                        elif self.selectedAudio[i] < -1:
                            self.selectedAudio[i] = -1
                elif option == 1:
                    if max(self.selectedAudio) > 1:
                        self.selectedAudio = self.selectedAudio/max(abs(self.selectedAudio))
                    elif min(self.selectedAudio) < -1:
                        self.selectedAudio = self.selectedAudio/min(abs(self.selectedAudio))
                rax._radio = radio
                
            radio.on_clicked(exceed)