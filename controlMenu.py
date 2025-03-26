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
    

    #####################
    # CALCULATE METHODS # taken from the original file: controlMenu.py
    #####################

    def calculateWaveform(self, ax):
        ax.plot(self.time, self.audio)
        ax.axhline(y=0, color='black', linewidth='0.5', linestyle='--') # draw an horizontal line in y=0.0
        ax.set(xlim=[0, self.duration], xlabel='Time (s)', ylabel='Amplitude')

    
    def calculateSTFT(self, audioFragWindow, nfft):
        stft = np.fft.fft(audioFragWindow, nfft)
        return stft[range(int(nfft/2))]
    
    
    def calculateWindowedSpectrogram(self, cm, ax, window, windSizeSampInt, hopSize, cmap):
        nfftUser = cm.var_nfft.get()
        draw = cm.var_draw.get()
        minfreq = cm.var_minf.get()
        maxfreq = cm.var_maxf.get()

        # Calculate the linear/mel spectrogram
        if draw == 1: # linear
            linear = librosa.stft(self.audio, n_fft=nfftUser, hop_length=hopSize, win_length=windSizeSampInt, window=window, center=True, dtype=None, pad_mode='constant')
            linear_dB = librosa.amplitude_to_db(np.abs(linear), ref=np.max)
            img = librosa.display.specshow(linear_dB, x_axis='time', y_axis='linear', sr=self.fs, fmin=minfreq, fmax=maxfreq, ax=ax, hop_length=hopSize, cmap=cmap)
            ax.set(ylim=[minfreq, maxfreq])
        else: # mel
            mel = librosa.feature.melspectrogram(y=self.audio, sr=self.fs, win_length=windSizeSampInt, n_fft=nfftUser, window=window, fmin=minfreq, fmax=maxfreq, hop_length=hopSize)
            mel_dB = librosa.power_to_db(mel)
            img = librosa.display.specshow(mel_dB, x_axis='time', y_axis='mel', sr=self.fs, fmin=minfreq, fmax=maxfreq, ax=ax, hop_length=hopSize, cmap=cmap)
            ax.set(ylim=[minfreq, maxfreq])
        self.yticks(minfreq, maxfreq) # represent the numbers of y axis
        
        return img
    

    def calculateSpectrogram(self, audio, ax, minfreq, maxfreq, draw, cmap):
        # Calculate the filtered linear/mel spectrogram filtered
        if draw == 1: # linear
            linear = librosa.stft(audio, center=True, dtype=None, pad_mode='constant')
            linear_dB = librosa.amplitude_to_db(np.abs(linear), ref=np.max)
            img = librosa.display.specshow(linear_dB, x_axis='time', y_axis='linear', sr=self.fs, fmin=minfreq, fmax=maxfreq, ax=ax, cmap=cmap)
            ax.set(xlim=[0, self.duration], ylim=[minfreq, maxfreq])
        else: # mel
            mel = librosa.feature.melspectrogram(y=audio, sr=self.fs, fmin=minfreq, fmax=maxfreq)
            mel_dB = librosa.power_to_db(mel)
            img = librosa.display.specshow(mel_dB, x_axis='time', y_axis='mel', sr=self.fs, fmin=minfreq, fmax=maxfreq, ax=ax, cmap=cmap)
            ax.set(xlim=[0, self.duration], ylim=[minfreq, maxfreq])

        return img
    
    
    def calculateSC(self, audioFragWindow):
        magnitudes = np.abs(np.fft.rfft(audioFragWindow)) # magnitudes of positive frequencies
        length = len(audioFragWindow)
        freqs = np.abs(np.fft.fftfreq(length, 1.0/self.fs)[:length//2+1]) # positive frequencies
        return np.sum(magnitudes*freqs)/np.sum(magnitudes) # return weighted mean
    
    
    def calculateSTE(self, sig, win, windSizeSampInt):
        window1 = signal.get_window(win, windSizeSampInt)
        window = window1 / len(window1)
        return signal.convolve(sig**2, window**2, mode='same')
    
    
    def calculatePitch(self, method, minpitch, maxpitch, maxCandidates):
        # Convert the numpy array containing the audio fragment into a wav file
        write('wav/frag.wav', self.fs, self.audio) # generates a wav file in the current folder

        silenceTh, voiceTh, octaveCost, octJumpCost, vcdUnvcdCost, accurate = self.controller.adse.getAutocorrelationVars()
        if accurate == 1: accurate_bool = True
        else: accurate_bool = False

        # Calculate the pitch of the generated wav file using parselmouth
        snd = parselmouth.Sound('wav/frag.wav')
        if method == 'Autocorrelation':
            pitch = snd.to_pitch_ac(pitch_floor=minpitch,
                                    max_number_of_candidates=maxCandidates,
                                    very_accurate=accurate_bool,
                                    silence_threshold=silenceTh,
                                    voicing_threshold=voiceTh,
                                    octave_cost=octaveCost,
                                    octave_jump_cost=octJumpCost,
                                    voiced_unvoiced_cost=vcdUnvcdCost,
                                    pitch_ceiling=maxpitch)
        elif method == 'Cross-correlation':
            pitch = snd.to_pitch_cc(pitch_floor=minpitch, 
                                    max_number_of_candidates=maxCandidates,
                                    very_accurate=accurate_bool,
                                    silence_threshold=silenceTh,
                                    voicing_threshold=voiceTh,
                                    octave_cost=octaveCost,
                                    octave_jump_cost=octJumpCost,
                                    voiced_unvoiced_cost=vcdUnvcdCost,
                                    pitch_ceiling=maxpitch)
        elif method == 'Subharmonics':
            maxFreqComp, maxSubharm, compFactor, pointsPerOct = self.controller.adse.getSubharmonicsVars()
            pitch = snd.to_pitch_shs(minimum_pitch=minpitch, 
                                    max_number_of_candidates=maxCandidates,
                                    maximum_frequency_component=maxFreqComp,
                                    max_number_of_subharmonics=maxSubharm,
                                    compression_factor=compFactor,
                                    ceiling=maxpitch,
                                    number_of_points_per_octave=pointsPerOct)
        elif method == 'Spinet':
            windLen, minFiltFreq, maxFiltFreq, numFilters = self.controller.adse.getSpinetVars()
            pitch = snd.to_pitch_spinet(window_length=windLen,
                                        minimum_filter_frequency=minFiltFreq,
                                        maximum_filter_frequency=maxFiltFreq,
                                        number_of_filters=numFilters,
                                        ceiling=maxpitch,
                                        max_number_of_candidates=maxCandidates)
        pitch_values = pitch.selected_array['frequency'] # extract selected pitch contour
        pitch_values[pitch_values==0] = np.nan # replace unvoiced samples by NaN to not plot

        return pitch, pitch_values
    
    
    def designFilter(self, cm, gpass, gstop):
        type = cm.var_pass.get() # harmonic, lowpass, highpass, bandpass or bandstop
        p = cm.var_perc.get()

        # Design filter
        if type == 'Lowpass' or type == 'Highpass':
            fcut = cm.var_fcut.get()
            delta = fcut * (p/100) # transition band

            if type == 'Lowpass':
                wp = fcut - delta
                ws = fcut + delta
            elif  type == 'Highpass':
                wp = fcut + delta
                ws = fcut - delta

            N, Wn = signal.ellipord(wp, ws, gpass, gstop, fs=self.fs)
            b, a = signal.ellip(N, gpass, gstop, Wn, btype=type, fs=self.fs)

        else:
            if type == 'Harmonic':
                fundfreqmult = cm.var_fund.get()
                fundfreq = cm.var_cent.get()
                fc = fundfreq * fundfreqmult # central frequency, value of the 1st harmonic
                fcut1 = fc - fundfreq/2
                fcut2 = fc + fundfreq/2
                delta1 = fcut1 * (p/100) # 1st transition band
                delta2 = fcut2 * (p/100) # 2nd transition band
                wp1 = fcut1 + delta1
                wp2 = fcut2 - delta2
                ws1 = fcut1 - delta1
                ws2 = fcut2 + delta2
            else:
                fcut1 = cm.var_cut1.get()
                fcut2 = cm.var_cut2.get()
                delta1 = fcut1 * (p/100) # 1st transition band
                delta2 = fcut2 * (p/100) # 2nd transition band

                if type == 'Bandpass':
                    wp1 = fcut1 + delta1
                    wp2 = fcut2 - delta2
                    ws1 = fcut1 - delta1
                    ws2 = fcut2 + delta2
                elif type == 'Bandstop':
                    wp1 = fcut1 - delta1
                    wp2 = fcut2 + delta2
                    ws1 = fcut1 + delta1
                    ws2 = fcut2 - delta2

            N, Wn = signal.ellipord([wp1,wp2], [ws1,ws2], gpass, gstop, fs=self.fs)
            b, a = signal.ellip(N, gpass, gstop, Wn, btype='Bandpass', fs=self.fs)

        filteredSignal = signal.lfilter(b, a, self.audio)

        return filteredSignal, b, a
    
    ################
    # PLOT METHODS # as in the original file
    ################
    
    # Plots the waveform and the Fast Fourier Transform (FFT) of the fragment
    def plotFT(self, cm):
        self.figFT, ax = plt.subplots(2, figsize=(12,6))
        self.figFT.suptitle('Fourier Transform')
        plt.subplots_adjust(hspace=.3) # to avoid overlapping between xlabel and title
        self.figFT.canvas.manager.set_window_title(self.fileName+'-FT')

        fft = np.fft.fft(self.audio) / self.lenAudio # Normalize amplitude
        fft2 = fft[range(int(self.lenAudio/2))] # Exclude sampling frequency
        values = np.arange(int(self.lenAudio/2))
        frequencies = values / (self.lenAudio/self.fs) # values / time period

        self.calculateWaveform(ax[0])
        ax[1].plot(frequencies, 20*np.log10(abs(fft2)))
        ax[1].set(xlim=[0, max(frequencies)], xlabel='Frequency (Hz)', ylabel='Amplitude (dB)')

        self.aux.saveasWavCsv(cm, self.figFT, self.time, self.audio, 0.5, self.fs) # save waveform as csv
        self.aux.saveasCsv(self.figFT, frequencies, 20*np.log10(abs(fft2)), 0.05, 'FT') # save FT as csv

        # TO-DO: connect figFrag with w1Button in signalVisualizer

        self.createSpanSelector(ax[0]) # Select a fragment with the cursor and play the audio of that fragment
        self.figFT.show() # show the figure

    

    def plotSTFT(self, cm, stft, frequencies, title):
        fig, ax = plt.subplots(2, figsize=(12,6))
        fig.suptitle('Short Time Fourier Transform')
        plt.subplots_adjust(hspace=.3) # to avoid overlapping between xlabel and title
        fig.canvas.manager.set_window_title(str(self.fileName)+'-STFT-'+title) # set title to the figure window

        self.calculateWaveform(ax[0])
        line1, = ax[1].plot(frequencies, 20*np.log10(abs(stft)))
        ax[1].set(xlim=[0, max(frequencies)], xlabel='Frequency (Hz)', ylabel='Amplitude (dB)')

        self.aux.saveasWavCsv(cm, fig, self.time, self.audio, 0.5, self.fs) # save waveform as csv
        self.aux.saveasCsv(fig, frequencies, 20*np.log10(abs(stft)), 0.05, 'STFT') # save FT as csv
        
        self.cursor = Cursor(ax[0], horizOn=False, useblit=True, color='black', linewidth=1)
        self.createSpanSelector(ax[0]) # Select a fragment with the cursor and play the audio of that fragment

        return ax, line1
    


    def plotSpectrogram(self, cm, window, windSizeSampInt, hopSize, cmap, title):
        showPitch = cm.var_pitch.get()

        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[1, 3])
        ax = gs.subplots(sharex=True)
        fig.suptitle('Spectrogram')
        fig.canvas.manager.set_window_title(str(self.fileName)+'-Spectrogram-'+title) # set title to the figure window

        # Hide x labels and tick labels for all but bottom plot.
        for a in ax:
            a.label_outer()
        
        # Calculate the linear/mel spectrogram
        img = self.calculateWindowedSpectrogram(cm, ax[1], window, windSizeSampInt, hopSize, cmap)
        self.colorBar(fig, 0.56, img)

        if showPitch == 1:
            method = cm.var_meth.get()
            minpitch = cm.var_minp.get()
            maxpitch = cm.var_maxp.get()
            maxCandidates, drawStyle = self.controller.adse.getVariables()
            pitch, pitch_values = self.calculatePitch(method, minpitch, maxpitch, maxCandidates)
            if drawStyle == 1: draw = '-'
            else: draw = 'o'
            ax[1].plot(pitch.xs(), pitch_values, draw, color='w')

        self.calculateWaveform(ax[0])
        self.aux.saveasWavCsv(cm, fig, self.time, self.audio, 0.69, self.fs) # save waveform as csv

        self.multicursor = MultiCursor(fig.canvas, (ax[0], ax[1]), color='black', lw=1)
        self.createSpanSelector(ax[0]) # Select a fragment with the cursor and play the audio of that fragment
        plt.show() # show the figure



    def plotSTFTspect(self, cm, stft, frequencies, window, windSizeSampInt, hopSize, cmap, title):
        fig = plt.figure(figsize=(12,6))
        ax1 = plt.subplot(311) # waveform
        ax2 = plt.subplot(312) # stft
        ax3 = plt.subplot(313, sharex=ax1) # spectrogram
        plt.subplots_adjust(hspace=.4) # to avoid overlapping between xlabel and title
        fig.suptitle('STFT + Spectrogram')
        fig.canvas.manager.set_window_title(str(self.fileName)+'-STFT+Spectrogram-'+title) # set title to the figure window

        self.calculateWaveform(ax1)

        line1, = ax2.plot(frequencies, 20*np.log10(abs(stft)))
        ax2.set(xlim=[0, max(frequencies)], xlabel='Frequency (Hz)', ylabel='Amplitude (dB)')

        # Calculate the linear/mel spectrogram
        img = self.calculateWindowedSpectrogram(cm, ax3, window, windSizeSampInt, hopSize, cmap)
        self.colorBar(fig, 0.17, img)

        self.aux.saveasWavCsv(cm, fig, self.time, self.audio, 0.65, self.fs) # save waveform as csv
        self.aux.saveasCsv(fig, frequencies, 20*np.log10(abs(stft)), 0.35, 'STFT') # save STFT as csv
        
        self.multicursor = MultiCursor(fig.canvas, (ax1, ax3), color='black', lw=1)
        self.createSpanSelector(ax1) # Select a fragment with the cursor and play the audio of that fragment

        return ax1, ax2, ax3, line1



    def plotSC(self, cm, audioFragWind2, window, windSizeSampInt, nfftUser, overlapSamp, hopSize, cmap, title):
        fig = plt.figure(figsize=(12,6))
        ax1 = plt.subplot(311) # waveform
        ax2 = plt.subplot(312) # power spectral density
        ax3 = plt.subplot(313, sharex=ax1) # spectrogram with spectral centroid
        plt.subplots_adjust(hspace=.6) # to avoid overlapping between xlabel and title
        fig.suptitle('Spectral Centroid')
        fig.canvas.manager.set_window_title(str(self.fileName)+'-SpectralCentroid-'+title) # set title to the figure window

        # Calculate the spectral centroid in the FFT as a vertical line
        spectralC = self.calculateSC(audioFragWind2)
        scValue = str(round(spectralC, 2)) # take only two decimals

        # Calculate the spectral centroid in the log power linear/mel spectrogram
        sc = librosa.feature.spectral_centroid(y=self.audio, sr=self.fs, n_fft=nfftUser, hop_length=hopSize, window=window, win_length=windSizeSampInt)
        times = librosa.times_like(sc, sr=self.fs, hop_length=hopSize, n_fft=nfftUser)
        
        self.calculateWaveform(ax1)

        _, freqs = ax2.psd(audioFragWind2, NFFT=windSizeSampInt, pad_to=nfftUser, Fs=self.fs, window=window, noverlap=overlapSamp)
        ax2.axvline(x=spectralC, color='r', linewidth='1') # draw a vertical line in x=value of the spectral centroid
        ax2.set(xlim=[0, max(freqs)], xlabel='Frequency (Hz)', ylabel='Power spectral density (dB/Hz)', title='Power spectral density using fft, spectral centroid value is '+ scValue)

        # Calculate the linear/mel spectrogram and the spectral centroid
        img = self.calculateWindowedSpectrogram(cm, ax3, window, windSizeSampInt, hopSize, cmap)
        self.colorBar(fig, 0.17, img)
        line1, = ax3.plot(times, sc.T, color='w') # draw the white line (sc)
        ax3.set(xlim=[0, self.duration], title='log Power spectrogram')

        self.aux.saveasWavCsv(cm, fig, self.time, self.audio, 0.65, self.fs) # save waveform as csv
        self.aux.saveasCsv(fig, times, sc.T, 0.35, 'SC') # save the white line as csv
        
        self.multicursor = MultiCursor(fig.canvas, (ax1, ax3), color='black', lw=1)
        self.createSpanSelector(ax1) # Select a fragment with the cursor and play the audio of that fragment

        return ax1, ax2, ax3, line1
    


    def plotSTE(self, cm, windType1, windSizeSampInt, title):
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(2, hspace=0)
        ax = gs.subplots(sharex=True)
        fig.suptitle('Short Time Energy')
        fig.canvas.manager.set_window_title(str(self.fileName)+'-STE-'+title) # set title to the figure window

        # Hide x labels and tick labels for all but bottom plot.
        for a in ax:
            a.label_outer()

        # Calculate the Short-Time-Energy
        signal = np.array(self.audio, dtype=float)
        time = np.arange(len(signal)) * (1.0/self.fs)
        ste = self.calculateSTE(signal, windType1, windSizeSampInt)

        self.calculateWaveform(ax[0])
        ax[1].plot(time, ste)
        ax[1].set(xlim=[0, self.duration], xlabel='Time (s)', ylabel='Amplitude (dB)')

        self.aux.saveasWavCsv(cm, fig, self.time, self.audio, 0.5, self.fs) # save waveform as csv
        self.aux.saveasCsv(fig, time, ste, 0.05, 'STE') # save STE as csv

        self.multicursor = MultiCursor(fig.canvas, (ax[0], ax[1]), color='black', lw=1)
        self.createSpanSelector(ax[0]) # Select a fragment with the cursor and play the audio of that fragment
        plt.show() # show the figure



    def plotPitch(self, cm):
        method = cm.var_meth.get()
        minpitch = cm.var_minp.get()
        maxpitch = cm.var_maxp.get()
        maxCandidates, drawStyle = self.controller.adse.getVariables()
        
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(2, hspace=0)
        ax = gs.subplots(sharex=True)
        fig.suptitle('Pitch measurement overtime')
        fig.canvas.manager.set_window_title('Pitch-Method_'+ str(method) +'-PitchFloor_'+ str(minpitch) + 'Hz-PitchCeiling_'+ str(maxpitch) + 'Hz') # set title to the figure window

        # Hide x labels and tick labels for all but bottom plot.
        for a in ax:
            a.label_outer()

        pitch, pitch_values = self.calculatePitch(method, minpitch, maxpitch, maxCandidates)

        if drawStyle == 1: draw = '-'
        else: draw = 'o'

        self.calculateWaveform(ax[0])
        ax[1].plot(pitch.xs(), pitch_values, draw)
        ax[1].set(xlim=[0, self.duration], xlabel='Time (s)', ylabel='Frequency (Hz)')

        self.aux.saveasWavCsv(cm, fig, self.time, self.audio, 0.5, self.fs) # save waveform as csv
        self.aux.saveasCsv(fig, pitch.xs(), pitch_values, 0.05, 'Pitch') # save Pitch as csv        

        self.multicursor = MultiCursor(fig.canvas, (ax[0], ax[1]), color='black', lw=1)
        self.createSpanSelector(ax[0]) # Select a fragment with the cursor and play the audio of that fragment
        plt.show() # show the figure


    def plotFiltering(self, cm):
        filteredSignal, _, _ = self.designFilter(cm, 3, 40)
        ControlMenu().createControlMenu(self.fileName+str(' (filtered)'), self.fs, filteredSignal, self.duration, self.controller)
        plt.show() # show the figure


    def closeEvent(self, event):
        plt.close('all')
        event.accept()