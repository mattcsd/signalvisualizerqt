import sys
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector, Button

from PyQt5.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QSpinBox, QPushButton, QGroupBox,  # Added QGroupBox here
    QMessageBox
)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QIcon

from auxiliar import Auxiliar
from controlMenu import ControlMenu
from help import Help

class FreeAdditionPureTones(QDialog):

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.fs = 48000  # sample frequency
        self.aux = Auxiliar()
        self.selectedAudio = np.empty(1)

        self.help = Help(self)  # Initialize help system like in PureTone

        self.pianoOpen = False
        self.amp_sliders = []
        self.freq_spinboxes = []
        
        # Default values
        self.default_values = [
            'Free addition of pure tones',
            'duration', '1.0',
            'octave', '4',
            'freq1', '440', 'freq2', '880', 'freq3', '1320',
            'freq4', '1760', 'freq5', '2200', 'freq6', '2640',
            'amp1', '1.0', 'amp2', '0.83', 'amp3', '0.67',
            'amp4', '0.5', 'amp5', '0.33', 'amp6', '0.17'
        ]
        
        # Load from CSV or use defaults
        try:
            csv_data = self.aux.readFromCsv()
            if len(csv_data) > 4 and len(csv_data[4]) >= 28:
                self.default_values = csv_data[4]
        except Exception as e:
            print(f"Error loading defaults: {e}")

        self.initUI()
        self.plotFAPT()

    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Control panel
        control_panel = QGroupBox("Tone Controls")
        control_layout = QGridLayout()
        control_panel.setLayout(control_layout)
        
        # Compact controls layout
        for i in range(6):
            # Frequency controls
            control_layout.addWidget(QLabel(f"Frq{i+1}"), 0, i*2)
            sb = QSpinBox()
            sb.setRange(0, 24000)
            sb.setValue(float(self.default_values[6+i*2]))
            sb.setMaximumWidth(80)
            self.freq_spinboxes.append(sb)
            control_layout.addWidget(sb, 1, i*2)
            
            # Amplitude controls
            control_layout.addWidget(QLabel(f"Amp{i+1}"), 0, i*2+1)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(float(self.default_values[18+i*2]) * 100))
            slider.setMaximumWidth(100)
            self.amp_sliders.append(slider)
            control_layout.addWidget(slider, 1, i*2+1)
            
            # Value display
            value_label = QLabel(f"{slider.value()/100:.2f}")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setMaximumWidth(40)
            slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(f"{v/100:.2f}"))
            control_layout.addWidget(value_label, 2, i*2+1)

        # Duration row
        dur_layout = QHBoxLayout()
        dur_layout.addWidget(QLabel('Duration (s):'))
        
        self.dur_slider = QSlider(Qt.Horizontal)
        self.dur_slider.setRange(1, 3000)
        self.dur_slider.setValue(int(float(self.default_values[2]) * 100))
        dur_layout.addWidget(self.dur_slider)
        
        self.dur_spinbox = QSpinBox()
        self.dur_spinbox.setRange(1, 3000)
        self.dur_spinbox.setValue(int(float(self.default_values[2]) * 100))
        self.dur_spinbox.setMaximumWidth(80)
        dur_layout.addWidget(self.dur_spinbox)
        
        # Buttons
        btn_layout = QHBoxLayout()
        buttons = [
            ('Plot', self.plotFAPT),
            ('Piano', self.togglePiano),
            ('Save', self.saveDefaultValues),
            ('🛈 Help', self.showHelp)
        ]
        
        for text, callback in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            btn.setMaximumWidth(80)
            btn_layout.addWidget(btn)
        
        # Add to main layout
        main_layout.addWidget(control_panel)
        main_layout.addLayout(dur_layout)
        main_layout.addLayout(btn_layout)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        
        self.setLayout(main_layout)
    def showHelp(self):
        """Show help window for this module"""
        if hasattr(self, 'help') and self.help:
            try:
                # Help ID 2 corresponds to "Addition of tones" help
                self.help.createHelpMenu(1)  
            except Exception as e:
                QMessageBox.warning(self, "Help Error", 
                                   f"Could not open help: {str(e)}")
        else:
            QMessageBox.information(self, "Help", 
                                   "Addition of Pure Tones Help\n\n"
                                   "This module lets you combine up to 6 pure tones.\n"
                                   "- Set frequencies (Hz) for each tone\n"
                                   "- Adjust amplitudes (0-1)\n"
                                   "- Control total duration\n"
                                   "- Use piano keyboard to set harmonic frequencies")

    def on_close(self):
        """Clean up resources when window closes"""
        if self.pianoOpen:
            self.piano.close()
        plt.close(self.fig)
        
    def getFrequencies(self):
        return [sb.value() for sb in self.freq_spinboxes]
    
    def getAmplitudes(self):
        return [slider.value() / 100 for slider in self.amp_sliders]
    
    def getDuration(self):
        return self.dur_slider.value() / 100
    
    def saveDefaultValues(self):
        try:
            # Get current values
            duration = self.getDuration()
            octave = self.octave_spinbox.value()
            freqs = self.getFrequencies()
            amps = self.getAmplitudes()
            
            # Create new values list
            new_values = [
                'Free addition of pure tones',
                'duration', str(duration),
                'octave', str(octave)
            ]
            
            # Add frequencies
            for i, freq in enumerate(freqs, start=1):
                new_values.extend([f'freq{i}', str(freq)])
            
            # Add amplitudes
            for i, amp in enumerate(amps, start=1):
                new_values.extend([f'amp{i}', str(amp)])
            
            # Read existing CSV data
            csv_data = self.aux.readFromCsv()
            if len(csv_data) < 5:
                # Pad with empty lists if needed
                csv_data.extend([[] for _ in range(5 - len(csv_data))])
            
            # Update our section (index 4)
            csv_data[4] = new_values
            
            # Save back to CSV
            self.aux.saveDefaultAsCsv(csv_data)
            QMessageBox.information(self, "Saved", "Default values saved successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not save values: {str(e)}")
    
    def showPiano(self):
        self.piano = QDialog(self)
        self.piano.setWindowTitle("Piano Keyboard")
        self.piano.setWindowIcon(QIcon('icons/icon.ico'))
        self.piano.setFixedSize(1000, 350)  # Larger window for more keys
        
        main_layout = QVBoxLayout()
        self.piano.setLayout(main_layout)
        
        # Octave controls at top
        octave_layout = QHBoxLayout()
        octave_layout.addWidget(QLabel("Octave:"))
        
        self.octave_spinbox = QSpinBox()
        self.octave_spinbox.setRange(1, 8)  # Now supports 8 octaves
        self.octave_spinbox.setValue(4)  # Middle C (C4) as default
        octave_layout.addWidget(self.octave_spinbox)
        
        main_layout.addLayout(octave_layout)
        
        # Piano keys area
        keys_widget = QWidget()
        keys_layout = QGridLayout()
        keys_widget.setLayout(keys_layout)
        
        # Piano key configuration - 2 octaves shown at once
        white_notes = [
            ('C', 0), ('D', 2), ('E', 4), ('F', 5), ('G', 7), ('A', 9), ('B', 11),
            ('C', 12), ('D', 14), ('E', 16), ('F', 17), ('G', 19), ('A', 21), ('B', 23)
        ]
        
        black_notes = [
            ('C#', 1), ('D#', 3), None, ('F#', 6), ('G#', 8), ('A#', 10),
            ('C#', 13), ('D#', 15), None, ('F#', 18), ('G#', 20), ('A#', 22)
        ]
        
        # Create white keys
        for i, (note, value) in enumerate(white_notes):
            btn = QPushButton(f"{note}\n{self.get_note_name(value)}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    border: 1px solid #ccc;
                    min-width: 50px;
                    min-height: 200px;
                    font-weight: bold;
                    font-size: 12px;
                    qproperty-alignment: 'AlignBottom|AlignHCenter';
                }
                QPushButton:pressed {
                    background-color: #ddd;
                }
            """)
            btn.clicked.connect(lambda _, v=value: self.playNote(v))
            keys_layout.addWidget(btn, 1, i*2, 2, 2)
        
        # Create black keys
        for i, key_info in enumerate(black_notes):
            if key_info is not None:
                note, value = key_info
                btn = QPushButton(f"{note}\n{self.get_note_name(value)}")
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: black;
                        color: white;
                        min-width: 36px;
                        min-height: 130px;
                        margin-right: 15px;
                        margin-left: 15px;
                        font-weight: bold;
                        font-size: 11px;
                        qproperty-alignment: 'AlignBottom|AlignHCenter';
                    }
                    QPushButton:pressed {
                        background-color: #555;
                    }
                """)
                btn.clicked.connect(lambda _, v=value: self.playNote(v))
                keys_layout.addWidget(btn, 0, (i*2)+1, 1, 2)
        
        main_layout.addWidget(keys_widget)
        
        # Add navigation buttons for different octave ranges
        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("◄ Previous Octave")
        prev_btn.clicked.connect(lambda: self.octave_spinbox.setValue(max(1, self.octave_spinbox.value()-1)))
        next_btn = QPushButton("Next Octave ►")
        next_btn.clicked.connect(lambda: self.octave_spinbox.setValue(min(8, self.octave_spinbox.value()+1)))
        
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        main_layout.addLayout(nav_layout)
        
        self.pianoOpen = True
        self.piano.show()

    def get_note_name(self, note_value):
        """Convert MIDI note number to scientific pitch notation (e.g., C4)"""
        octave = (note_value // 12) + self.octave_spinbox.value()
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return f"{note_names[note_value % 12]}{octave}"

    def playNote(self, note_value):
        """Play a note and set harmonics based on current octave"""
        octave = self.octave_spinbox.value()
        midi_note = note_value + (octave * 12)
        frequency = 440 * (2 ** ((midi_note - 69) / 12))  # A4 = 440Hz = MIDI 69
        
        # Set fundamental frequency
        self.freq_spinboxes[0].setValue(round(frequency, 2))
        self.amp_sliders[0].setValue(100)  # Full amplitude
        
        # Set harmonics
        harmonics = [(2, 0.83), (3, 0.67), (4, 0.5), (5, 0.33), (6, 0.17)]
        for i, (multiple, amp) in enumerate(harmonics, start=1):
            self.freq_spinboxes[i].setValue(round(frequency * multiple, 2))
            self.amp_sliders[i].setValue(int(amp * 100))
        
        # Play the note
        duration = 1.0  # seconds
        samples = int(duration * self.fs)
        time = np.linspace(0, duration, samples, endpoint=False)
        signal = np.sin(2 * np.pi * frequency * time)
        sd.play(signal, self.fs)
        
        self.plotFAPT()
    def notesHarmonics(self, note_value):
        """Handle piano key presses with the new note numbering"""
        octave = self.octave_spinbox.value()
        # Convert to frequency (A4 = 440Hz is note 69 in MIDI)
        fundfreq = 440 * (2 ** ((note_value - 69 + (octave-4)*12) / 12))
        
        # Set fundamental frequency
        self.freq_spinboxes[0].setValue(round(fundfreq, 2))
        self.amp_sliders[0].setValue(100)  # 1.0
        
        # Set harmonics (2nd through 6th)
        harmonics = [(2, 0.83), (3, 0.67), (4, 0.5), (5, 0.33), (6, 0.17)]
        for i, (multiple, amp) in enumerate(harmonics, start=1):
            freq = fundfreq * multiple
            self.freq_spinboxes[i].setValue(round(freq, 2))
            self.amp_sliders[i].setValue(int(amp * 100))
        
        self.plotFAPT()
    
    def togglePiano(self):
        if not self.pianoOpen:
            self.showPiano()
        else:
            self.piano.close()
    
    

    def plotFAPT(self):
        duration = self.getDuration()
        samples = int(duration * self.fs)
        freqs = self.getFrequencies()
        amps = self.getAmplitudes()
        
        time = np.linspace(0, duration, samples, endpoint=False)
        signal = np.zeros(samples)
        
        for freq, amp in zip(freqs, amps):
            signal += amp * np.sin(2 * np.pi * freq * time)
        
        self.ax.clear()
        self.ax.plot(time, signal)
        
        # Configure plot
        limit = max(abs(signal)) * 1.1 if max(abs(signal)) > 0 else 1.1
        self.ax.set(
            xlim=[0, duration],
            ylim=[-limit, limit],
            xlabel='Time (s)',
            ylabel='Amplitude'
        )
        self.ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        self.ax.grid(True)
        
        # Add load button
        self.addLoadButton(signal, duration)
        
        self.canvas.draw()
    
    def addLoadButton(self, audio, duration):
        # Remove previous button if exists
        if hasattr(self, 'load_btn_ax'):
            self.load_btn_ax.remove()
        
        # Create new button
        self.load_btn_ax = self.fig.add_axes([0.8, 0.01, 0.09, 0.05])
        self.load_btn = Button(self.load_btn_ax, 'Load')
        
        def load(event):
            if self.selectedAudio.shape == (1,):
                self.cm = ControlMenu()
                self.cm.createControlMenu(
                    'Free addition of pure tones',
                    self.fs,
                    audio,
                    duration,
                    self.controller
                )
            else:
                durSelec = len(self.selectedAudio) / self.fs
                self.cm = ControlMenu()
                self.cm.createControlMenu(
                    'Free addition of pure tones',
                    self.fs,
                    self.selectedAudio,
                    durSelec,
                    self.controller
                )
            self.close()
        
        self.load_btn.on_clicked(load)
        
        # Span selector for audio playback
        time = np.linspace(0, duration, len(audio), endpoint=False)
        
        def onselect(xmin, xmax):
            ini, end = np.searchsorted(time, (xmin, xmax))
            self.selectedAudio = audio[ini:end+1]
            sd.play(self.selectedAudio, self.fs)
        
        self.span = SpanSelector(
            self.ax,
            onselect,
            'horizontal',
            useblit=True,
            interactive=True,
            drag_from_anywhere=True
        )
        
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FreeAdditionPureTones(None)
    window.show()
    sys.exit(app.exec_())