from PyQt5 import QtWidgets, QtGui
import sys
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy import signal

from auxiliar import Auxiliar
from controlMenu import ControlMenu

class SquareWave(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.aux = Auxiliar()
        self.cm = ControlMenu()
        self.fig, self.ax = plt.subplots()
        self.selectedAudio = np.empty(1)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Generate Square Wave')
        self.setGeometry(100, 100, 850, 500)

        layout = QtWidgets.QVBoxLayout()
        
        # Read default values from CSV
        data = self.aux.readFromCsv()
        duration, amplitude, self.fs, offset, frequency, phase, cycle = (
            data[2][2], data[2][4], data[2][6], data[2][8], data[2][10], data[2][12], data[2][14]
        )
        
        # Sliders
        self.sliders = {}
        params = {'Duration (s)': duration, 'Offset': offset, 'Amplitude': amplitude, 
                  'Frequency (Hz)': frequency, 'Phase (π rad)': phase, 'Active Cycle (%)': cycle}
        
        for key, val in params.items():
            label = QtWidgets.QLabel(key)
            slider = QtWidgets.QSlider()
            slider.setOrientation(QtCore.Qt.Horizontal)
            slider.setValue(int(val))
            layout.addWidget(label)
            layout.addWidget(slider)
            self.sliders[key] = slider

        # Buttons
        self.plotButton = QtWidgets.QPushButton('Plot')
        self.plotButton.clicked.connect(self.plotSquareWave)
        layout.addWidget(self.plotButton)

        self.setLayout(layout)

    def plotSquareWave(self):
        amplitude = self.sliders['Amplitude'].value()
        frequency = self.sliders['Frequency (Hz)'].value()
        phase = self.sliders['Phase (π rad)'].value()
        cycle = self.sliders['Active Cycle (%)'].value()
        duration = self.sliders['Duration (s)'].value()
        offset = self.sliders['Offset'].value()
        samples = int(duration * self.fs)

        time = np.linspace(0, duration, samples, endpoint=False)
        square = amplitude * (signal.square(2 * np.pi * frequency * time + phase * np.pi, duty=cycle / 100) / 2) + offset
        
        self.ax.clear()
        self.ax.plot(time, square)
        self.ax.set(title='Square Wave', xlabel='Time (s)', ylabel='Amplitude')
        plt.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = SquareWave()
    window.show()
    sys.exit(app.exec_())
