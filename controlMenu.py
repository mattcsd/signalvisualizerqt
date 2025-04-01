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

from PyQt5.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton, QRadioButton, 
    QCheckBox, QComboBox, QGridLayout, QMessageBox, QHBoxLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5 import QtGui

from auxiliar import Auxiliar
from pitchAdvancedSettings import AdvancedSettings

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


    def updateOptionMenu(self, menu):
        """Update the options in a QComboBox menu"""
        current_text = menu.currentText()
        menu.clear()
        menu.addItems([str(x) for x in self.opt_nfft])
        # Try to restore the previous selection if it still exists
        index = menu.findText(current_text)
        if index >= 0:
            menu.setCurrentIndex(index)

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
            self.updateOptionMenu(self.dd_nfft)  # This was the line with the error
            
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
        try:
            minfreq = float(self.ent_minf.text())
            maxfreq = float(self.ent_maxf.text())
            
            if minfreq >= maxfreq:
                self.ent_minf.setText('0')
                text = f"The minimum frequency must be smaller than the maximum frequency ({maxfreq}Hz)."
                QMessageBox.critical(self, "Minimum frequency too big", text)
                return False
            return True
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Please enter a valid number for minimum frequency")
            return False

    def maxfreqEntry(self):
        try:
            minfreq = float(self.ent_minf.text())
            maxfreq = float(self.ent_maxf.text())
            
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
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Please enter a valid number for maximum frequency")
            return False


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


    def plotFigure(self, choice, windSize, overlap, minfreq, maxfreq, beta):
        """Handle plotting based on the selected option"""
        try:
            # Store current values
            self.var_opts = choice
            self.var_size = windSize
            self.var_over = overlap
            self.var_minf = minfreq
            self.var_maxf = maxfreq
            self.var_beta = beta
            
            # Calculate common parameters
            windSizeSamp = int(windSize * self.fs)
            overlapSamp = int(overlap * self.fs)
            hopSize = windSizeSamp - overlapSamp
            nfftUser = int(self.dd_nfft.currentText())
            
            # Get window type
            windType = self.dd_wind.currentText().lower()
            window = signal.get_window(windType, windSizeSamp)
            
            # Handle each plot type
            if choice == 'FT':
                self.plotFT()
            elif choice == 'STFT':
                stft = self.calculateSTFT(window, nfftUser)
                freqs = np.fft.fftfreq(nfftUser, 1.0/self.fs)[:nfftUser//2]
                self.plotSTFT(stft, freqs, windType)
            elif choice == 'Spectrogram':
                self.plotSpectrogram(window, windSizeSamp, hopSize, 'viridis', windType)
            elif choice == 'STFT + Spect':
                stft = self.calculateSTFT(window, nfftUser)
                freqs = np.fft.fftfreq(nfftUser, 1.0/self.fs)[:nfftUser//2]
                self.plotSTFTspect(stft, freqs, window, windSizeSamp, hopSize, 'viridis', windType)
            elif choice == 'Short-Time-Energy':
                self.plotSTE(windType, windSizeSamp, windType)
            elif choice == 'Pitch':
                self.plotPitch()
            elif choice == 'Spectral Centroid':
                audioFragWind2 = self.audio[:windSizeSamp] * window
                self.plotSC(audioFragWind2, window, windSizeSamp, nfftUser, overlapSamp, hopSize, 'viridis', windType)
            elif choice == 'Filtering':
                self.plotFiltering()
                
        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"Failed to create plot: {str(e)}")

    def checkValues(self):
        try:
            choice = self.dd_opts.currentText()
            windSize = float(self.ent_size.text())
            overlap = float(self.ent_over.text())
            
            # Handle frequency inputs
            minfreq = float(self.ent_minf.text())
            maxfreq = float(self.ent_maxf.text())
            minfreq = int(minfreq) if minfreq.is_integer() else minfreq
            maxfreq = int(maxfreq) if maxfreq.is_integer() else maxfreq
            
            beta = int(self.ent_beta.text())
            minpitch = float(self.ent_minp.text())
            maxpitch = float(self.ent_maxp.text())
            fundfreq = int(self.ent_fund.text())
            center = int(self.ent_cent.text())
            percentage = float(self.ent_perc.text())
            fcut1 = int(self.ent_cut1.text())
            fcut2 = int(self.ent_cut2.text())

            # Validate inputs
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
            
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid input value: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")
    


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
    
    def yticks(self, minfreq, maxfreq):
        """Set appropriate y-axis ticks for spectrogram plots"""
        # Create evenly spaced frequency ticks
        if maxfreq <= 1000:
            step = 100
        elif maxfreq <= 5000:
            step = 500
        else:
            step = 1000
        
        ticks = np.arange(0, maxfreq + step, step)
        ticks = ticks[ticks >= minfreq]
        plt.yticks(ticks)
    
    def calculateWindowedSpectrogram(self, ax, window, windSizeSampInt, hopSize, cmap):
        nfftUser = self.var_nfft
        draw = self.var_draw
        minfreq = self.var_minf
        maxfreq = self.var_maxf
        
        # Ensure nfftUser is at least as large as the window size
        if nfftUser < windSizeSampInt:
            nfftUser = windSizeSampInt
            self.dd_nfft.setCurrentText(str(nfftUser))
            self.var_nfft = nfftUser

        try:
            if draw == 1: # linear
                linear = librosa.stft(self.audio, 
                                    n_fft=nfftUser, 
                                    hop_length=hopSize, 
                                    win_length=windSizeSampInt, 
                                    window=window)
                linear_dB = librosa.amplitude_to_db(np.abs(linear), ref=np.max)
                img = librosa.display.specshow(linear_dB, 
                                             x_axis='time', 
                                             y_axis='linear', 
                                             sr=self.fs, 
                                             fmin=minfreq, 
                                             fmax=maxfreq, 
                                             ax=ax, 
                                             hop_length=hopSize, 
                                             cmap=cmap)
                ax.set(ylim=[minfreq, maxfreq])
            else: # mel
                mel = librosa.feature.melspectrogram(y=self.audio, 
                                                   sr=self.fs, 
                                                   n_fft=nfftUser,
                                                   hop_length=hopSize,
                                                   win_length=windSizeSampInt,
                                                   window=window,
                                                   fmin=minfreq,
                                                   fmax=maxfreq)
                mel_dB = librosa.power_to_db(mel, ref=np.max)
                img = librosa.display.specshow(mel_dB, 
                                             x_axis='time', 
                                             y_axis='mel', 
                                             sr=self.fs, 
                                             fmin=minfreq, 
                                             fmax=maxfreq, 
                                             ax=ax, 
                                             hop_length=hopSize, 
                                             cmap=cmap)
                ax.set(ylim=[minfreq, maxfreq])
            
            # Either remove this line or implement the yticks method
            # self.yticks(minfreq, maxfreq)
            
            return img
            
        except Exception as e:
            QMessageBox.critical(self, "Spectrogram Error", 
                               f"Failed to create spectrogram: {str(e)}")
            return None

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
    
    
    def designFilter(self, gpass, gstop):
        type = self.var_pass # harmonic, lowpass, highpass, bandpass or bandstop
        p = self.var_perc

        # Design filter
        if type == 'Lowpass' or type == 'Highpass':
            fcut = self.var_fcut
            delta = fcut * (p/100)

            if type == 'Lowpass':
                wp = fcut - delta
                ws = fcut + delta
            elif type == 'Highpass':
                wp = fcut + delta
                ws = fcut - delta

            N, Wn = signal.ellipord(wp, ws, gpass, gstop, fs=self.fs)
            b, a = signal.ellip(N, gpass, gstop, Wn, btype=type, fs=self.fs)
        else:
            if type == 'Harmonic':
                fundfreqmult = self.var_fund
                fundfreq = self.var_cent
                fc = fundfreq * fundfreqmult
                fcut1 = fc - fundfreq/2
                fcut2 = fc + fundfreq/2
                delta1 = fcut1 * (p/100)
                delta2 = fcut2 * (p/100)
                wp1 = fcut1 + delta1
                wp2 = fcut2 - delta2
                ws1 = fcut1 - delta1
                ws2 = fcut2 + delta2
            else:
                fcut1 = self.var_cut1
                fcut2 = self.var_cut2
                delta1 = fcut1 * (p/100)
                delta2 = fcut2 * (p/100)

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


    def colorBar(self, fig, position, img):
        """Add a colorbar to the figure with proper positioning"""
        try:
            # Create axes for colorbar
            cax = fig.add_axes([0.85, position, 0.02, 0.2])  # [left, bottom, width, height]
            cbar = fig.colorbar(img, cax=cax)
            cbar.set_label('dB')
            return cbar
        except Exception as e:
            QMessageBox.critical(self, "Colorbar Error", 
                               f"Failed to create colorbar: {str(e)}")
            return None
        
    ################
    # PLOT METHODS # as in the original file
    ################

    def plotFT(self):
        self.figFT, ax = plt.subplots(2, figsize=(12,6))
        self.figFT.suptitle('Fourier Transform')
        plt.subplots_adjust(hspace=.3)
        self.figFT.canvas.manager.set_window_title(self.fileName+'-FT')

        fft = np.fft.fft(self.audio) / self.lenAudio
        fft2 = fft[range(int(self.lenAudio/2))]
        values = np.arange(int(self.lenAudio/2))
        frequencies = values / (self.lenAudio/self.fs)

        self.calculateWaveform(ax[0])
        ax[1].plot(frequencies, 20*np.log10(abs(fft2)))
        ax[1].set(xlim=[0, max(frequencies)], xlabel='Frequency (Hz)', ylabel='Amplitude (dB)')

        self.aux.saveasWavCsv(self, self.figFT, self.time, self.audio, 0.5, self.fs)
        self.aux.saveasCsv(self.figFT, frequencies, 20*np.log10(abs(fft2)), 0.05, 'FT')

        self.createSpanSelector(ax[0])
        self.figFT.show()

    def plotSTFT(self, stft, frequencies, title):
        fig, ax = plt.subplots(2, figsize=(12,6))
        fig.suptitle('Short Time Fourier Transform')
        plt.subplots_adjust(hspace=.3)
        fig.canvas.manager.set_window_title(f"{self.fileName}-STFT-{title}")

        self.calculateWaveform(ax[0])
        line1, = ax[1].plot(frequencies, 20*np.log10(abs(stft)))
        ax[1].set(xlim=[0, max(frequencies)], xlabel='Frequency (Hz)', ylabel='Amplitude (dB)')

        self.aux.saveasWavCsv(self, self.figFT, self.time, self.audio, 0.5, self.fs)
        self.aux.saveasCsv(self, fig, frequencies, 20*np.log10(abs(stft)), 0.05, 'STFT')
        
        self.cursor = Cursor(ax[0], horizOn=False, useblit=True, color='black', linewidth=1)
        self.createSpanSelector(ax[0])
        return ax, line1


    def plotSpectrogram(self, window, windSizeSampInt, hopSize, cmap, title):
        showPitch = self.var_pitch

        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[1, 3])
        ax = gs.subplots(sharex=True)
        fig.suptitle('Spectrogram')
        fig.canvas.manager.set_window_title(f"{self.fileName}-Spectrogram-{title}")

        for a in ax:
            a.label_outer()
        
        img = self.calculateWindowedSpectrogram( ax[1], window, windSizeSampInt, hopSize, cmap)
        self.colorBar(fig, 0.56, img)

        if showPitch == 1:
            method = self.var_meth
            minpitch = self.var_minp
            maxpitch = self.var_maxp
            maxCandidates, drawStyle = self.controller.adse.getVariables()
            pitch, pitch_values = self.calculatePitch(method, minpitch, maxpitch, maxCandidates)
            draw = '-' if drawStyle == 1 else 'o'
            ax[1].plot(pitch.xs(), pitch_values, draw, color='w')

        self.calculateWaveform(ax[0])
        self.aux.saveasWavCsv(self, fig, self.time, self.audio, 0.69, self.fs)

        self.multicursor = MultiCursor(fig.canvas, (ax[0], ax[1]), color='black', lw=1)
        self.createSpanSelector(ax[0])
        plt.show()

    def plotSTFTspect(self, stft, frequencies, window, windSizeSampInt, hopSize, cmap, title):
        fig = plt.figure(figsize=(12,6))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313, sharex=ax1)
        plt.subplots_adjust(hspace=.4)
        fig.suptitle('STFT + Spectrogram')
        fig.canvas.manager.set_window_title(f"{self.fileName}-STFT+Spectrogram-{title}")

        self.calculateWaveform(ax1)

        line1, = ax2.plot(frequencies, 20*np.log10(abs(stft)))
        ax2.set(xlim=[0, max(frequencies)], xlabel='Frequency (Hz)', ylabel='Amplitude (dB)')

        img = self.calculateWindowedSpectrogram( ax3, window, windSizeSampInt, hopSize, cmap)
        self.colorBar(fig, 0.17, img)

        self.aux.saveasWavCsv(self, fig, self.time, self.audio, 0.65, self.fs)
        self.aux.saveasCsv(fig, frequencies, 20*np.log10(abs(stft)), 0.35, 'STFT')
        
        self.multicursor = MultiCursor(fig.canvas, (ax1, ax3), color='black', lw=1)
        self.createSpanSelector(ax1)
        return ax1, ax2, ax3, line1

    def plotSC(self, audioFragWind2, window, windSizeSampInt, nfftUser, overlapSamp, hopSize, cmap, title):
        fig = plt.figure(figsize=(12,6))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313, sharex=ax1)
        plt.subplots_adjust(hspace=.6)
        fig.suptitle('Spectral Centroid')
        fig.canvas.manager.set_window_title(f"{self.fileName}-SpectralCentroid-{title}")

        spectralC = self.calculateSC(audioFragWind2)
        scValue = str(round(spectralC, 2))

        sc = librosa.feature.spectral_centroid(y=self.audio, sr=self.fs, n_fft=nfftUser, 
                                             hop_length=hopSize, window=window, win_length=windSizeSampInt)
        times = librosa.times_like(sc, sr=self.fs, hop_length=hopSize, n_fft=nfftUser)
        
        self.calculateWaveform(ax1)

        _, freqs = ax2.psd(audioFragWind2, NFFT=windSizeSampInt, pad_to=nfftUser, 
                          Fs=self.fs, window=window, noverlap=overlapSamp)
        ax2.axvline(x=spectralC, color='r', linewidth='1')
        ax2.set(xlim=[0, max(freqs)], xlabel='Frequency (Hz)', 
               ylabel='Power spectral density (dB/Hz)', 
               title=f'Power spectral density using fft, spectral centroid value is {scValue}')

        img = self.calculateWindowedSpectrogram( ax3, window, windSizeSampInt, hopSize, cmap)
        self.colorBar(fig, 0.17, img)
        line1, = ax3.plot(times, sc.T, color='w')
        ax3.set(xlim=[0, self.duration], title='log Power spectrogram')

        self.aux.saveasWavCsv(self, fig, self.time, self.audio, 0.65, self.fs)
        self.aux.saveasCsv(fig, times, sc.T, 0.35, 'SC')
        
        self.multicursor = MultiCursor(fig.canvas, (ax1, ax3), color='black', lw=1)
        self.createSpanSelector(ax1)
        return ax1, ax2, ax3, line1

    def plotSTE(self, windType1, windSizeSampInt, title):
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(2, hspace=0)
        ax = gs.subplots(sharex=True)
        fig.suptitle('Short Time Energy')
        fig.canvas.manager.set_window_title(f"{self.fileName}-STE-{title}")

        for a in ax:
            a.label_outer()

        signal = np.array(self.audio, dtype=float)
        time = np.arange(len(signal)) * (1.0/self.fs)
        ste = self.calculateSTE(signal, windType1, windSizeSampInt)

        self.calculateWaveform(ax[0])
        ax[1].plot(time, ste)
        ax[1].set(xlim=[0, self.duration], xlabel='Time (s)', ylabel='Amplitude (dB)')

        self.aux.saveasWavCsv(self, fig, self.time, self.audio, 0.5, self.fs)
        self.aux.saveasCsv(fig, time, ste, 0.05, 'STE')

        self.multicursor = MultiCursor(fig.canvas, (ax[0], ax[1]), color='black', lw=1)
        self.createSpanSelector(ax[0])
        plt.show()

    def plotPitch(self):
        method = self.var_meth
        minpitch = self.var_minp
        maxpitch = self.var_maxp
        maxCandidates, drawStyle = self.controller.adse.getVariables()
        
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(2, hspace=0)
        ax = gs.subplots(sharex=True)
        fig.suptitle('Pitch measurement overtime')
        fig.canvas.manager.set_window_title(
            f'Pitch-Method_{method}-PitchFloor_{minpitch}Hz-PitchCeiling_{maxpitch}Hz')

        for a in ax:
            a.label_outer()

        pitch, pitch_values = self.calculatePitch(method, minpitch, maxpitch, maxCandidates)
        draw = '-' if drawStyle == 1 else 'o'

        self.calculateWaveform(ax[0])
        ax[1].plot(pitch.xs(), pitch_values, draw)
        ax[1].set(xlim=[0, self.duration], xlabel='Time (s)', ylabel='Frequency (Hz)')

        self.aux.saveasWavCsv(self, fig, self.time, self.audio, 0.5, self.fs)
        self.aux.saveasCsv(fig, pitch.xs(), pitch_values, 0.05, 'Pitch')

        self.multicursor = MultiCursor(fig.canvas, (ax[0], ax[1]), color='black', lw=1)
        self.createSpanSelector(ax[0])
        plt.show()

    def plotFiltering(self):
        filteredSignal, _, _ = self.designFilter(self, 3, 40)
        ControlMenu(fileName=f"{self.fileName} (filtered)", fs=self.fs, 
                   audioFrag=filteredSignal, duration=self.duration, 
                   controller=self.controller, parent=self).show()
        plt.show()

    def closeEvent(self, event):
        plt.close('all')
        event.accept()
