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
        self.aux = Auxiliar()
        self.cm = ControlMenu()
        self.selectedAudio = np.empty(1)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.setupUI()
        
    def setupUI(self):
        self.setWindowTitle('Generate pure tone')
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(850, 475)
        
        # Read default values
        list = self.aux.readFromCsv()
        duration = list[1][2]
        amplitude = list[1][4]
        self.fs = list[1][6]
        offset = list[1][8]
        frequency = list[1][10]
        phase = list[1][12]
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Matplotlib figure
        fig_layout = QVBoxLayout()
        fig_layout.addWidget(self.toolbar)
        fig_layout.addWidget(self.canvas)
        main_layout.addLayout(fig_layout)
        
        # Control layout
        control_layout = QGridLayout()
        control_layout.setSpacing(10)
        
        # Variables
        self.var_dura = duration
        self.var_offs = offset
        self.var_ampl = amplitude
        self.var_freq = frequency
        self.var_phas = phase
        self.var_fs = self.fs
        
        # Sliders
        self.sca_dura = QSlider(Qt.Horizontal)
        self.sca_dura.setRange(1, 3000)  # 0.01 to 30.00 in steps of 0.01
        self.sca_dura.setValue(int(duration * 100))
        self.sca_offs = QSlider(Qt.Horizontal)
        self.sca_offs.setRange(-100, 100)  # -1.00 to 1.00 in steps of 0.01
        self.sca_offs.setValue(int(offset * 100))
        self.sca_ampl = QSlider(Qt.Horizontal)
        self.sca_ampl.setRange(0, 100)  # 0.00 to 1.00 in steps of 0.01
        self.sca_ampl.setValue(int(amplitude * 100))
        self.sca_freq = QSlider(Qt.Horizontal)
        self.sca_freq.setRange(0, 24000)  # 0 to 24000 Hz
        self.sca_freq.setValue(frequency)
        self.sca_phas = QSlider(Qt.Horizontal)
        self.sca_phas.setRange(-100, 100)  # -1.00 to 1.00 in steps of 0.01
        self.sca_phas.setValue(int(phase * 100))
        
        # Connect slider signals
        self.sca_dura.valueChanged.connect(self.updateDuration)
        self.sca_offs.valueChanged.connect(self.updateOffset)
        self.sca_ampl.valueChanged.connect(self.updateAmplitude)
        self.sca_freq.valueChanged.connect(self.updateFrequency)
        self.sca_phas.valueChanged.connect(self.updatePhase)
        
        # Add sliders to layout
        control_layout.addWidget(QLabel('Total duration (s)'), 0, 0)
        control_layout.addWidget(self.sca_dura, 0, 1, 1, 3)
        control_layout.addWidget(QLabel('Offset'), 1, 0)
        control_layout.addWidget(self.sca_offs, 1, 1, 1, 3)
        control_layout.addWidget(QLabel('Amplitude'), 2, 0)
        control_layout.addWidget(self.sca_ampl, 2, 1, 1, 3)
        control_layout.addWidget(QLabel('Frequency (Hz)'), 3, 0)
        control_layout.addWidget(self.sca_freq, 3, 1, 1, 3)
        control_layout.addWidget(QLabel(f'Phase (π rad)'), 4, 0)
        control_layout.addWidget(self.sca_phas, 4, 1, 1, 3)
        
        # Entries
        self.ent_dura = QLineEdit(f"{duration:.2f}")
        self.ent_offs = QLineEdit(f"{offset:.2f}")
        self.ent_ampl = QLineEdit(f"{amplitude:.2f}")
        self.ent_freq = QLineEdit(str(frequency))
        self.ent_phas = QLineEdit(f"{phase:.2f}")
        self.ent_fs = QLineEdit(str(self.fs))
        
        # Connect entry signals
        self.ent_dura.editingFinished.connect(self.entryDurationChanged)
        self.ent_offs.editingFinished.connect(self.entryOffsetChanged)
        self.ent_ampl.editingFinished.connect(self.entryAmplitudeChanged)
        self.ent_freq.editingFinished.connect(self.entryFrequencyChanged)
        self.ent_phas.editingFinished.connect(self.entryPhaseChanged)
        self.ent_fs.editingFinished.connect(self.entryFsChanged)
        
        # Add entries to layout
        control_layout.addWidget(self.ent_dura, 0, 4)
        control_layout.addWidget(self.ent_offs, 1, 4)
        control_layout.addWidget(self.ent_ampl, 2, 4)
        control_layout.addWidget(self.ent_freq, 3, 4)
        control_layout.addWidget(self.ent_phas, 4, 4)
        
        # Expression label
        self.lab_sign = QLabel()
        self.updateExpression()
        control_layout.addWidget(QLabel('Expression'), 5, 0)
        control_layout.addWidget(self.lab_sign, 5, 1, 1, 3)
        control_layout.addWidget(QLabel('Fs (Hz)'), 5, 3)
        control_layout.addWidget(self.ent_fs, 5, 4)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.but_save = QPushButton('Save')
        self.but_save.clicked.connect(self.saveDefaultValues)
        self.but_plot = QPushButton('Plot')
        self.but_plot.clicked.connect(self.plotPureTone)
        self.but_help = QPushButton('🛈')
        self.but_help.setFixedWidth(30)
        self.but_help.clicked.connect(lambda: self.controller.help.createHelpMenu(1))
        
        button_layout.addWidget(self.but_save)
        button_layout.addWidget(self.but_plot)
        button_layout.addWidget(self.but_help)
        control_layout.addLayout(button_layout, 6, 1, 1, 4)
        
        main_layout.addLayout(control_layout)
        self.setLayout(main_layout)
        
        # Initial plot
        self.plotPureTone()
    
    def updateExpression(self):
        pi_symbol = unicodedata.lookup("GREEK SMALL LETTER PI")
        sign = f"{self.var_offs:.2f} + {self.var_ampl:.2f} COS(2{pi_symbol} {self.var_freq}t + {self.var_phas:.2f}{pi_symbol})"
        self.lab_sign.setText(sign)
    
    # Slider update methods
    def updateDuration(self, value):
        self.var_dura = value / 100
        self.ent_dura.setText(f"{self.var_dura:.2f}")
        self.updateExpression()
    
    def updateOffset(self, value):
        self.var_offs = value / 100
        self.ent_offs.setText(f"{self.var_offs:.2f}")
        self.updateExpression()
    
    def updateAmplitude(self, value):
        self.var_ampl = value / 100
        self.ent_ampl.setText(f"{self.var_ampl:.2f}")
        self.updateExpression()
    
    def updateFrequency(self, value):
        self.var_freq = value
        self.ent_freq.setText(str(self.var_freq))
        self.updateExpression()
    
    def updatePhase(self, value):
        self.var_phas = value / 100
        self.ent_phas.setText(f"{self.var_phas:.2f}")
        self.updateExpression()
    
    # Entry update methods
    def entryDurationChanged(self):
        try:
            value = float(self.ent_dura.text())
            if 0.01 <= value <= 30:
                self.var_dura = value
                self.sca_dura.setValue(int(value * 100))
            else:
                self.ent_dura.setText(f"{self.var_dura:.2f}")
        except ValueError:
            self.ent_dura.setText(f"{self.var_dura:.2f}")
    
    def entryOffsetChanged(self):
        try:
            value = float(self.ent_offs.text())
            if -1 <= value <= 1:
                self.var_offs = value
                self.sca_offs.setValue(int(value * 100))
            else:
                self.ent_offs.setText(f"{self.var_offs:.2f}")
        except ValueError:
            self.ent_offs.setText(f"{self.var_offs:.2f}")
    
    def entryAmplitudeChanged(self):
        try:
            value = float(self.ent_ampl.text())
            if 0 <= value <= 1:
                self.var_ampl = value
                self.sca_ampl.setValue(int(value * 100))
            else:
                self.ent_ampl.setText(f"{self.var_ampl:.2f}")
        except ValueError:
            self.ent_ampl.setText(f"{self.var_ampl:.2f}")
    
    def entryFrequencyChanged(self):
        try:
            value = int(self.ent_freq.text())
            if 0 <= value <= 24000:
                self.var_freq = value
                self.sca_freq.setValue(value)
            else:
                self.ent_freq.setText(str(self.var_freq))
        except ValueError:
            self.ent_freq.setText(str(self.var_freq))
    
    def entryPhaseChanged(self):
        try:
            value = float(self.ent_phas.text())
            if -1 <= value <= 1:
                self.var_phas = value
                self.sca_phas.setValue(int(value * 100))
            else:
                self.ent_phas.setText(f"{self.var_phas:.2f}")
        except ValueError:
            self.ent_phas.setText(f"{self.var_phas:.2f}")
    
    def entryFsChanged(self):
        try:
            value = int(self.ent_fs.text())
            if value > 48000:
                self.ent_fs.setText(str(self.fs))
                QMessageBox.critical(self, 'Wrong sample frequency value', 
                                   'The sample frequency cannot be greater than 48000 Hz.')
            else:
                self.fs = value
        except ValueError:
            self.ent_fs.setText(str(self.fs))
    
    def saveDefaultValues(self):
        list = self.aux.readFromCsv()
        new_list = [
            ['NOISE','\t duration', list[0][2],'\t amplitude', list[0][4],'\t fs', list[0][6],'\t noise type', list[0][8]],
            ['PURE TONE','\t duration', self.var_dura,'\t amplitude', self.var_ampl,'\t fs', self.fs,
             '\t offset', self.var_offs,'\t frequency', self.var_freq,'\t phase', self.var_phas],
            ['SQUARE WAVE','\t duration', list[2][2],'\t amplitude', list[2][4],'\t fs', list[2][6],
             '\t offset', list[2][8],'\t frequency', list[2][10],'\t phase', list[2][12],'\t active cycle', list[2][14]],
            ['SAWTOOTH WAVE','\t duration', list[3][2],'\t amplitude', list[3][4],'\t fs', list[3][6],
             '\t offset', list[3][8],'\t frequency', list[3][10],'\t phase', list[3][12],'\t max position', list[3][14]],
            ['FREE ADD OF PT','\t duration', list[4][2],'\t octave', list[4][4],'\t freq1', list[4][6],
             '\t freq2', list[4][8],'\t freq3', list[4][10],'\t freq4', list[4][12],'\t freq5', list[4][14],
             '\t freq6', list[4][16],'\t amp1', list[4][18],'\t amp2', list[4][20],'\t amp3', list[4][22],
             '\t amp4', list[4][24],'\t amp5', list[4][26],'\t amp6', list[4][28]],
            ['SPECTROGRAM','\t colormap', list[5][2]]
        ]
        self.aux.saveDefaultAsCsv(new_list)
    
    def plotPureTone(self):
        # Check frequency
        self.aux.bigFrequency(self.var_freq, self.fs)
        
        # Generate signal
        samples = int(self.var_dura * self.fs)
        time = np.linspace(start=0, stop=self.var_dura, num=samples, endpoint=False)
        ptone = self.var_ampl * (np.cos(2*np.pi * self.var_freq*time + self.var_phas*np.pi)) + self.var_offs
        
        # Clear and redraw
        self.ax.clear()
        
        # Plot the pure tone
        limite = max(abs(ptone))*1.1
        self.ax.plot(time, ptone)
        self.fig.canvas.setWindowTitle('Pure tone')
        self.ax.set(xlim=[0, self.var_dura], ylim=[-limite, limite], 
                   xlabel='Time (s)', ylabel='Amplitude')
        self.ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        self.ax.axhline(y=1.0, color='red', linewidth=0.8, linestyle='--')
        self.ax.axhline(y=-1.0, color='red', linewidth=0.8, linestyle='--')
        self.ax.axhline(y=self.var_offs, color='blue', linewidth=1, label="offset")
        self.ax.legend(loc="upper right")
        
        # Add load button
        self.addLoadButton(self.fig, self.ax, self.fs, time, ptone, self.var_dura, 'Pure tone')
        
        # Add scale/saturate radio buttons if needed
        if self.var_offs > 0.5 or self.var_offs < -0.5:
            self.addScaleSaturateRadiobuttons(self.fig, self.var_offs)
        
        self.canvas.draw()
    
    def addLoadButton(self, fig, ax, fs, time, audio, duration, name):
        axload = fig.add_axes([0.8, 0.01, 0.09, 0.05])
        but_load = Button(axload, 'Load')
        
        def load(event):
            if self.selectedAudio.shape == (1,): 
                self.cm.createControlMenu(name, fs, audio, duration, self.controller)
            else:
                time = np.arange(0, len(self.selectedAudio)/fs, 1/fs)
                durSelec = max(time)
                self.cm.createControlMenu(name, fs, self.selectedAudio, durSelec, self.controller)
            plt.close(fig)
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
        rax = fig.add_axes([0.75, 0.9, 0.15, 0.1])
        radio = RadioButtons(rax, ('scale', 'saturate'))
        
        def exceed(label):
            options = {'scale': 0, 'saturate': 1}
            option = options[label]
            if option == 0:  # scale
                for i in range(len(self.selectedAudio)):
                    if self.selectedAudio[i] > 1:
                        self.selectedAudio[i] = 1
                    elif self.selectedAudio[i] < -1:
                        self.selectedAudio[i] = -1
            elif option == 1:  # saturate
                if max(self.selectedAudio) > 1:
                    self.selectedAudio = self.selectedAudio/max(abs(self.selectedAudio))
                elif min(self.selectedAudio) < -1:
                    self.selectedAudio = self.selectedAudio/min(abs(self.selectedAudio))
            rax._radio = radio
        
        radio.on_clicked(exceed)