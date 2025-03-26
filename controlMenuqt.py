import math
import librosa, librosa.display 
import parselmouth
import numpy as np
import sounddevice as sd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.widgets import Cursor, SpanSelector, MultiCursor
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
from scipy.io.wavfile import write

from auxiliar import Auxiliar
from pitchAdvancedSettings import AdvancedSettings

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDialog, QLabel, QLineEdit, QPushButton, 
                             QRadioButton, QCheckBox, QComboBox, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal

class ControlMenu(QDialog):
    def __init__(self, fileName, fs, audioFrag, duration, controller, parent=None):
        super().__init__(parent)
        self.fileName = fileName
        self.audio = audioFrag  # audio array of the fragment
        self.fs = fs  # sample frequency of the audio (Hz)
        self.time = np.arange(0, len(self.audio)/self.fs, 1/self.fs)  # time array of the audio
        self.duration = duration  # duration of the audio (s)
        self.lenAudio = len(self.audio)  # length of the audio array
        self.controller = controller
        np.seterr(divide='ignore')  # turn off the "RuntimeWarning: divide by zero encountered in log10"

        # 'self.time' and 'self.audio' need to have the same first dimension
        if len(self.time) < len(self.audio):
            self.audio = self.audio[:-1].copy()  # delete last element of the numpy array
        elif len(self.time) > len(self.audio):
            self.time = self.time[:-1].copy()  # delete last element of the numpy array

        self.aux = Auxiliar()

        # The signal must have a minimum duration of 0.01 seconds
        if self.duration < 0.01:
            text = "The signal must have a minimum duration of 0.01s."
            QMessageBox.critical(self, "Signal too short", text)
            return

        self.setupUI()

    def setupUI(self):
        self.setWindowTitle(self.fileName)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(750, 575)
        
        # Main layout
        layout = QGridLayout()
        layout.setSpacing(10)
        
        # Options
        self.options = ['FT', 'STFT', 'Spectrogram', 'STFT + Spect', 'Short-Time-Energy', 
                       'Pitch', 'Spectral Centroid', 'Filtering']
        self.opt_wind = ['Bartlett', 'Blackman', 'Hamming', 'Hanning', 'Kaiser']
        self.opt_nfft = [2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19]
        self.opt_meth = ['Autocorrelation', 'Cross-correlation', 'Subharmonics', 'Spinet']
        self.opt_pass = ['Harmonic', 'Lowpass', 'Highpass', 'Bandpass', 'Bandstop']

        # Variables
        if self.duration <= 0.03:
            windSize = round(self.duration - 0.001, 3)
            overlap = round(windSize - 0.001, 3)
        else:
            windSize = 0.03
            overlap = 0.01

        self.var_size = windSize
        self.var_over = overlap
        self.var_minf = 0
        self.var_maxf = self.fs/2
        self.var_minp = 75.0
        self.var_maxp = 600.0
        self.var_fund = 1
        self.var_cent = 400
        self.var_perc = 10.0
        self.var_fcut = 1000
        self.var_cut1 = 200
        self.var_cut2 = 600
        self.var_beta = 0
        self.var_draw = 1
        self.var_pitch = 0
        self.var_opts = 'Spectrogram'
        self.var_wind = 'Hamming'
        self.var_nfft = self.opt_nfft[0]
        self.var_meth = 'Autocorrelation'
        self.var_pass = 'Harmonic'

        # Widgets
        # Labels
        layout.addWidget(QLabel('Choose an option', alignment=Qt.AlignRight), 0, 0, 1, 2)
        layout.addWidget(QLabel('Window'), 1, 0, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('nfft'), 3, 0, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('Method'), 11, 0, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('Filter type'), 2, 2, alignment=Qt.AlignRight)
        
        layout.addWidget(QLabel('Window length (s)'), 2, 0, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('Overlap (s)'), 4, 0, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('Min frequency (Hz)'), 6, 0, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('Max frequency (Hz)'), 7, 0, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('Pitch floor (Hz)'), 12, 0, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('Pitch ceiling (Hz)'), 13, 0, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('Fund. freq. multiplication'), 3, 2, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('First harmonic frequency'), 4, 2, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('Percentage (%)'), 5, 2, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('Fcut'), 6, 2, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('Fcut1'), 7, 2, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('Fcut2'), 8, 2, alignment=Qt.AlignRight)
        layout.addWidget(QLabel('Beta'), 10, 2, alignment=Qt.AlignRight)
        layout.addWidget(QLabel(f'Fs: {self.fs} Hz'), 13, 3, alignment=Qt.AlignRight)
        
        layout.addWidget(QLabel('Spectrogram', alignment=Qt.AlignCenter), 5, 1)
        layout.addWidget(QLabel('Pitch', alignment=Qt.AlignCenter), 10, 1)
        layout.addWidget(QLabel('Filtering', alignment=Qt.AlignCenter), 1, 3)
        layout.addWidget(QLabel('Short-Time-Energy', alignment=Qt.AlignCenter), 9, 3)
        layout.addWidget(QLabel('Drawing style'), 8, 0, alignment=Qt.AlignRight)

        # Entries
        self.ent_size = QLineEdit(str(self.var_size))
        self.ent_over = QLineEdit(str(self.var_over))
        self.ent_minf = QLineEdit(str(self.var_minf))
        self.ent_maxf = QLineEdit(str(self.var_maxf))
        self.ent_minp = QLineEdit(str(self.var_minp))
        self.ent_maxp = QLineEdit(str(self.var_maxp))
        self.ent_fund = QLineEdit(str(self.var_fund))
        self.ent_cent = QLineEdit(str(self.var_cent))
        self.ent_perc = QLineEdit(str(self.var_perc))
        self.ent_fcut = QLineEdit(str(self.var_fcut))
        self.ent_cut1 = QLineEdit(str(self.var_cut1))
        self.ent_cut2 = QLineEdit(str(self.var_cut2))
        self.ent_beta = QLineEdit(str(self.var_beta))

        # Set validators
        double_validator = QtGui.QDoubleValidator()
        int_validator = QtGui.QIntValidator()
        
        self.ent_size.setValidator(double_validator)
        self.ent_over.setValidator(double_validator)
        self.ent_minf.setValidator(int_validator)
        self.ent_maxf.setValidator(int_validator)
        self.ent_minp.setValidator(double_validator)
        self.ent_maxp.setValidator(double_validator)
        self.ent_fund.setValidator(int_validator)
        self.ent_cent.setValidator(int_validator)
        self.ent_perc.setValidator(double_validator)
        self.ent_fcut.setValidator(int_validator)
        self.ent_cut1.setValidator(int_validator)
        self.ent_cut2.setValidator(int_validator)
        self.ent_beta.setValidator(int_validator)

        # Connect returnPressed signals
        self.ent_size.returnPressed.connect(self.windowLengthEntry)
        self.ent_over.returnPressed.connect(self.overlapEntry)
        self.ent_minf.returnPressed.connect(self.minfreqEntry)
        self.ent_maxf.returnPressed.connect(self.maxfreqEntry)
        self.ent_minp.returnPressed.connect(self.minpitchEntry)
        self.ent_maxp.returnPressed.connect(self.maxpitchEntry)
        self.ent_fund.returnPressed.connect(self.fundfreqEntry)
        self.ent_cent.returnPressed.connect(self.centerEntry)
        self.ent_perc.returnPressed.connect(self.percentageEntry)
        self.ent_cut1.returnPressed.connect(self.fcut1Entry)
        self.ent_cut2.returnPressed.connect(self.fcut2Entry)
        self.ent_beta.returnPressed.connect(self.betaEntry)

        # Add entries to layout
        layout.addWidget(self.ent_size, 2, 1)
        layout.addWidget(self.ent_over, 4, 1)
        layout.addWidget(self.ent_minf, 6, 1)
        layout.addWidget(self.ent_maxf, 7, 1)
        layout.addWidget(self.ent_minp, 12, 1)
        layout.addWidget(self.ent_maxp, 13, 1)
        layout.addWidget(self.ent_fund, 3, 3)
        layout.addWidget(self.ent_cent, 4, 3)
        layout.addWidget(self.ent_perc, 5, 3)
        layout.addWidget(self.ent_fcut, 6, 3)
        layout.addWidget(self.ent_cut1, 7, 3)
        layout.addWidget(self.ent_cut2, 8, 3)
        layout.addWidget(self.ent_beta, 10, 3)

        # Radio buttons
        self.rdb_lin = QRadioButton('linear')
        self.rdb_mel = QRadioButton('mel')
        self.rdb_lin.setChecked(True)
        
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.rdb_lin)
        radio_layout.addWidget(self.rdb_mel)
        layout.addLayout(radio_layout, 8, 1)

        # Checkbox
        self.chk_pitch = QCheckBox('Show pitch')
        self.chk_pitch.stateChanged.connect(self.pitchCheckbox)
        layout.addWidget(self.chk_pitch, 9, 1, alignment=Qt.AlignLeft)

        # Buttons
        self.but_adse = QPushButton('Advanced settings')
        self.but_adse.setEnabled(False)
        self.but_adse.clicked.connect(lambda: self.controller.adse.advancedSettings())
        
        self.but_plot = QPushButton('Plot')
        self.but_plot.clicked.connect(self.checkValues)
        
        self.but_help = QPushButton('🛈')
        self.but_help.setFixedWidth(30)
        self.but_help.clicked.connect(lambda: self.controller.help.createHelpMenu(8))

        layout.addWidget(self.but_adse, 14, 1)
        layout.addWidget(self.but_plot, 14, 3)
        layout.addWidget(self.but_help, 14, 2, alignment=Qt.AlignRight)

        # Option menus
        self.dd_opts = QComboBox()
        self.dd_opts.addItems(self.options)
        self.dd_opts.setCurrentText(self.var_opts)
        self.dd_opts.currentTextChanged.connect(self.displayOptions)
        
        self.dd_wind = QComboBox()
        self.dd_wind.addItems(self.opt_wind)
        self.dd_wind.setCurrentText(self.var_wind)
        
        self.dd_nfft = QComboBox()
        self.dd_nfft.addItems([str(x) for x in self.opt_nfft])
        self.dd_nfft.setCurrentText(str(self.var_nfft))
        
        self.dd_meth = QComboBox()
        self.dd_meth.addItems(self.opt_meth)
        self.dd_meth.setCurrentText(self.var_meth)
        self.dd_meth.setEnabled(False)
        
        self.dd_pass = QComboBox()
        self.dd_pass.addItems(self.opt_pass)
        self.dd_pass.setCurrentText(self.var_pass)
        self.dd_pass.setEnabled(False)
        self.dd_pass.currentTextChanged.connect(self.displayFilterOptions)

        layout.addWidget(self.dd_opts, 0, 2)
        layout.addWidget(self.dd_wind, 1, 1)
        layout.addWidget(self.dd_nfft, 3, 1)
        layout.addWidget(self.dd_meth, 11, 1)
        layout.addWidget(self.dd_pass, 2, 3)

        # Set stretch factors
        for i in range(4):
            layout.setColumnStretch(i, 1)
        for i in range(15):
            layout.setRowStretch(i, 1)

        self.setLayout(layout)
        
        # Initialize widget states
        self.displayOptions(self.var_opts)
        self.displayFilterOptions(self.var_pass)

    # Methods for entry validation
    def windowLengthEntry(self):
        windSize = float(self.ent_size.text())
        overlap = float(self.ent_over.text())
        
        if windSize > self.duration or windSize == 0:
            if self.duration <= 0.03:
                self.var_size = round(self.duration - 0.001, 3)
            else:
                self.var_size = 0.03
            self.ent_size.setText(str(self.var_size))
            
            self.opt_nfft = [2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19]
            self.updateOptionMenu(self.dd_nfft)
            
            if windSize > self.duration:
                text = f"The window size can't be greater than the duration of the signal ({self.duration}s)."
                QMessageBox.critical(self, "Window size too long", text)
            elif windSize == 0:
                QMessageBox.critical(self, "Wrong window size value", "The chosen value for the window size must be a positive number.")
            return False
        elif windSize < overlap:
            self.var_size = overlap + 0.01
            self.ent_size.setText(str(self.var_size))
            text = f"The window size must always be greater than the overlap ({overlap}s)."
            QMessageBox.critical(self, "Wrong overlap value", text)
            return False
        else:
            windSizeSamp = windSize * self.fs
            nfft = self.opt_nfft[0]
            
            if nfft < windSizeSamp:
                last = int(math.log2(self.opt_nfft[-1])) + 1
                first = int(math.log2(nfft))
                while 2**first < windSizeSamp:
                    for a in range(len(self.opt_nfft)-1):
                        self.opt_nfft[a] = self.opt_nfft[a+1]
                    self.opt_nfft[-1] = 2**last
                    last += 1
                    first += 1
                self.updateOptionMenu(self.dd_nfft)
            else:
                first = int(math.log2(nfft)) - 1
                while 2**first > windSizeSamp:
                    for a in range(len(self.opt_nfft)-1, 0, -1):
                        self.opt_nfft[a] = self.opt_nfft[a-1]
                    self.opt_nfft[0] = 2**first
                    self.updateOptionMenu(self.dd_nfft)
                    first -= 1
            return True

    def overlapEntry(self):
        overlap = float(self.ent_over.text())
        windSize = float(self.ent_size.text())
        
        if overlap > self.duration or overlap >= windSize:
            if self.duration <= 0.03:
                overlap = round(windSize - 0.001, 3)
            else:
                overlap = 0.01
            self.ent_over.setText(str(overlap))
            
            if overlap > self.duration:
                text = f"The overlap can't be greater than the duration of the signal ({self.duration}s)."
                QMessageBox.critical(self, "Overlap too long", text)
            elif overlap >= windSize:
                text = f"The overlap must always be smaller than the window size ({windSize}s)."
                QMessageBox.critical(self, "Wrong overlap value", text)
            return False
        return True

    def minfreqEntry(self):
        minfreq = int(self.ent_minf.text())
        maxfreq = int(self.ent_maxf.text())
        
        if minfreq >= maxfreq:
            self.ent_minf.setText('0')
            text = f"The minimum frequency must be smaller than the maximum frequency ({maxfreq}Hz)."
            QMessageBox.critical(self, "Minimum frequency too big", text)
            return False
        return True

    def maxfreqEntry(self):
        minfreq = int(self.ent_minf.text())
        maxfreq = int(self.ent_maxf.text())
        
        if maxfreq > self.fs/2 or maxfreq <= minfreq:
            self.ent_maxf.setText(str(int(self.fs/2)))
            
            if maxfreq > self.fs/2:
                text = f"The maximum frequency can't be greater than the half of the sample frequency ({self.fs/2}Hz)."
                QMessageBox.critical(self, "Maximum frequency too big", text)
            elif maxfreq <= minfreq:
                text = f"The maximum frequency must be greater than the minimum frequency ({minfreq}Hz)."
                QMessageBox.critical(self, "Maximum frequency too small", text)
            return False
        return True

    def minpitchEntry(self):
        minPitch = float(self.ent_minp.text())
        maxPitch = float(self.ent_maxp.text())
        
        if minPitch >= maxPitch:
            self.ent_minp.setText('75.0')
            self.ent_maxp.setText('600.0')
            text = f"The minimum pitch must be smaller than the maximum pitch ({maxPitch}Hz)."
            QMessageBox.critical(self, "Pitch floor too big", text)
            return False
        return True

    def maxpitchEntry(self):
        minPitch = float(self.ent_minp.text())
        maxPitch = float(self.ent_maxp.text())
        
        if maxPitch <= minPitch:
            self.ent_minp.setText('75.0')
            self.ent_maxp.setText('600.0')
            text = f"The maximum pitch must be greater than the minimum pitch ({minPitch}Hz)."
            QMessageBox.critical(self, "Pitch ceiling too small", text)
            return False
        return True

    def fundfreqEntry(self):
        fundfreq = int(self.ent_fund.text())
        
        if fundfreq < 1:
            self.ent_fund.setText('1')
            text = "The minimum value of the fundamental frequency response is 1."
            QMessageBox.critical(self, "Fundamental frequency response too small", text)
            return False
        elif fundfreq > (self.fs/2):
            self.ent_fund.setText(str(int(self.fs/2)))
            text = f"The maximum frequency can't be greater than the half of the sample frequency ({self.fs/2}Hz)."
            QMessageBox.critical(self, "Fundamental frequency response too big", text)
            return False
        return True

    def centerEntry(self):
        return True

    def percentageEntry(self):
        percentage = float(self.ent_perc.text())
        
        if percentage < 0.0 or percentage > 100.0:
            self.ent_perc.setText('10.0')
            text = "The percentage must be a number between 0 and 100."
            QMessageBox.critical(self, "Wrong percentage value", text)
            return False
        return True

    def fcut1Entry(self):
        return True

    def fcut2Entry(self):
        return True

    def betaEntry(self):
        beta = int(self.ent_beta.text())
        
        if beta < 0 or beta > 14:
            self.ent_beta.setText('0')
            text = "The value of beta must be a number between 0 and 14."
            QMessageBox.critical(self, "Incorrect value of beta", text)
            return False
        return True

    def pitchCheckbox(self):
        showPitch = self.chk_pitch.isChecked()
        
        if showPitch:
            self.controller.adse.createVariables()
            self.dd_meth.setEnabled(True)
            self.ent_minp.setEnabled(True)
            self.ent_maxp.setEnabled(True)
            self.but_adse.setEnabled(True)
        else:
            self.dd_meth.setEnabled(False)
            self.ent_minp.setEnabled(False)
            self.ent_maxp.setEnabled(False)
            self.but_adse.setEnabled(False)

    def displayOptions(self, choice):
        self.dd_opts.setEnabled(True)
        
        # Enable/disable widgets based on choice
        self.ent_over.setEnabled(choice not in ['FT', 'STFT', 'Pitch', 'Filtering'])
        self.ent_size.setEnabled(choice not in ['FT', 'Pitch', 'Filtering'])
        
        if choice == 'Filtering':
            self.dd_pass.setEnabled(True)
            self.ent_perc.setEnabled(True)
            
            filter_type = self.dd_pass.currentText()
            self.ent_fcut.setEnabled(filter_type in ['Lowpass', 'Highpass'])
            self.ent_cut1.setEnabled(filter_type in ['Bandpass', 'Bandstop'])
            self.ent_cut2.setEnabled(filter_type in ['Bandpass', 'Bandstop'])
            self.ent_fund.setEnabled(filter_type == 'Harmonic')
            self.ent_cent.setEnabled(filter_type == 'Harmonic')
        else:
            self.dd_pass.setEnabled(False)
            self.ent_fund.setEnabled(False)
            self.ent_cent.setEnabled(False)
            self.ent_perc.setEnabled(False)
            self.ent_fcut.setEnabled(False)
            self.ent_cut1.setEnabled(False)
            self.ent_cut2.setEnabled(False)
        
        if choice == 'Pitch':
            self.controller.adse.createVariables()
            self.dd_meth.setEnabled(True)
            self.ent_minp.setEnabled(True)
            self.ent_maxp.setEnabled(True)
            self.but_adse.setEnabled(True)
        else:
            self.dd_meth.setEnabled(False)
            self.ent_minp.setEnabled(False)
            self.ent_maxp.setEnabled(False)
            self.but_adse.setEnabled(False)
        
        self.rdb_lin.setEnabled(choice in ['Spectrogram', 'STFT + Spect', 'Spectral Centroid', 'Filtering'])
        self.rdb_mel.setEnabled(choice in ['Spectrogram', 'STFT + Spect', 'Spectral Centroid', 'Filtering'])
        
        self.ent_minf.setEnabled(choice in ['Spectrogram', 'STFT + Spect', 'Spectral Centroid'])
        self.ent_maxf.setEnabled(choice in ['Spectrogram', 'STFT + Spect', 'Spectral Centroid'])
        
        self.dd_wind.setEnabled(choice in ['STFT', 'Spectrogram', 'STFT + Spect', 'Spectral Centroid', 'Short-Time-Energy'])
        self.dd_nfft.setEnabled(choice in ['STFT', 'Spectrogram', 'STFT + Spect', 'Spectral Centroid'])
        
        self.ent_beta.setEnabled(choice == 'Short-Time-Energy')
        self.chk_pitch.setEnabled(choice == 'Spectrogram')

    def displayFilterOptions(self, choice):
        self.ent_fcut.setEnabled(choice in ['Lowpass', 'Highpass'])
        self.ent_cut1.setEnabled(choice in ['Bandpass', 'Bandstop'])
        self.ent_cut2.setEnabled(choice in ['Bandpass', 'Bandstop'])
        self.ent_fund.setEnabled(choice == 'Harmonic')
        self.ent_cent.setEnabled(choice == 'Harmonic')

    def checkValues(self):
        choice = self.dd_opts.currentText()
        windSize = float(self.ent_size.text())
        overlap = float(self.ent_over.text())
        minfreq = int(self.ent_minf.text())
        maxfreq = int(self.ent_maxf.text())
        beta = int(self.ent_beta.text())
        minpitch = float(self.ent_minp.text())
        maxpitch = float(self.ent_maxp.text())
        fundfreq = int(self.ent_fund.text())
        center = int(self.ent_cent.text())
        percentage = float(self.ent_perc.text())
        fcut1 = int(self.ent_cut1.text())
        fcut2 = int(self.ent_cut2.text())

        if choice in ['STFT', 'STFT + Spect', 'Spectral Centroid', 'Spectrogram', 'Filtering', 'Short-Time-Energy']:
            if choice == 'Short-Time-Energy' and not self.betaEntry():
                return
            if not self.minfreqEntry() or not self.maxfreqEntry():
                return
            if choice == 'Filtering' and (not self.fundfreqEntry() or not self.centerEntry() or 
                                        not self.percentageEntry() or not self.fcut1Entry() or 
                                        not self.fcut2Entry()):
                return
            if choice != 'Filtering' and not self.windowLengthEntry():
                return
            if choice in ['STFT + Spect', 'Spectral Centroid', 'Short-Time-Energy', 'Spectrogram'] and not self.overlapEntry():
                return
        elif choice == 'Pitch' and (not self.minpitchEntry() or not self.maxpitchEntry()):
            return
        
        self.plotFigure(choice, windSize, overlap, minfreq, maxfreq, beta)

    # All the other methods (calculate* and plot*) remain exactly the same as in your original code
    # since they don't interact with the GUI directly
    
    # ... [rest of your methods unchanged] ...

    def closeEvent(self, event):
        plt.close('all')
        event.accept()