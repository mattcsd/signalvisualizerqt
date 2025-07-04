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

from queue import Queue
import threading

from auxiliar import Auxiliar
from controlMenu import ControlMenu
from help import Help

class FreeAdditionPureTones(QDialog):

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.fs = 48000  # sample frequency
        sd.default.samplerate = self.fs
        sd.default.blocksize = 1024  # Optimal for responsiveness
        sd.default.latency = 'low'
        self.aux = Auxiliar()
        self.selectedAudio = np.empty(1)

        self.help = Help(self)  # Initialize help system like in PureTone

        self.piano = None  # Initialize reference
        self.pianoOpen = False
        self.amp_sliders = []
        self.freq_spinboxes = []

        # Initialize audio storage
        self.full_audio = np.empty(1)  # Will store the complete generated audio
        self.selectedAudio = np.empty(1)  # Will store selected portions
        self.audio_duration = 0.0
        
        # Default values
        self.default_values = [
            'Free addition of pure tones',
            'duration', '0.3',
            'octave', '4',
            'freq1', '440', 'freq2', '880', 'freq3', '1320',
            'freq4', '1760', 'freq5', '2200', 'freq6', '2640',
            'amp1', '1.0', 'amp2', '0.83', 'amp3', '0.67',
            'amp4', '0.5', 'amp5', '0.33', 'amp6', '0.17'
        ]
        
        self.audio_queue = Queue()
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()

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

            int_dval = int(self.default_values[6+i*2])
            sb.setValue(int_dval)

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
            btn.setMaximumWidth(100 if text == 'Load to Controller' else 80)
            btn_layout.addWidget(btn)
            if text == 'Piano':  # Store reference to piano button
                self.piano_btn = btn
        
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
                self.help.createHelpMenu(2)  
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
        self.piano.setFixedSize(1000, 350)
        
        main_layout = QVBoxLayout()
        self.piano.setLayout(main_layout)
        
        # Octave controls at top
        octave_layout = QHBoxLayout()
        octave_layout.addWidget(QLabel("Octave:"))
        
        self.octave_spinbox = QSpinBox()
        self.octave_spinbox.setRange(1, 8)
        self.octave_spinbox.setValue(4)
        self.octave_spinbox.valueChanged.connect(self.update_piano_labels)  # Connect value change
        octave_layout.addWidget(self.octave_spinbox)
        
        # Store references to all piano keys
        self.piano_keys = []
        
        main_layout.addLayout(octave_layout)
        
        # Piano keys area
        keys_widget = QWidget()
        keys_layout = QGridLayout()
        keys_widget.setLayout(keys_layout)
        
        # Piano key configuration
        white_notes = [
            ('C', 0), ('D', 2), ('E', 4), ('F', 5), ('G', 7), ('A', 9), ('B', 11),
            ('C', 12), ('D', 14), ('E', 16), ('F', 17), ('G', 19), ('A', 21), ('B', 23)
        ]
        
        black_notes = [
            ('C#', 1), ('D#', 3), None, ('F#', 6), ('G#', 8), ('A#', 10),
            ('C#', 13), ('D#', 15), None, ('F#', 18), ('G#', 20), ('A#', 22)
        ]
        
        # Create white keys with black text
        for i, (note, value) in enumerate(white_notes):
            btn = QPushButton()
            btn.note_value = value
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    color: black;  /* Black text */
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
            btn.clicked.connect(lambda checked, v=value: self.playPianoNote(v))  # Modified connection
            keys_layout.addWidget(btn, 1, i*2, 2, 2)
            self.piano_keys.append(btn)
        
        # Create black keys
        for i, key_info in enumerate(black_notes):
            if key_info is not None:
                note, value = key_info
                btn = QPushButton()
                btn.note_value = value
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
                btn.clicked.connect(lambda checked, v=value: self.playPianoNote(v))  # Modified connection
                keys_layout.addWidget(btn, 0, (i*2)+1, 1, 2)
                self.piano_keys.append(btn)
        
        main_layout.addWidget(keys_widget)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("◄ Previous Octave")
        prev_btn.clicked.connect(self.decrease_octave)
        next_btn = QPushButton("Next Octave ►")
        next_btn.clicked.connect(self.increase_octave)
        
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        main_layout.addLayout(nav_layout)
        
        # Initial label update
        self.update_piano_labels()
        
        def on_close():
            self.piano = None
            if hasattr(self, 'piano_btn'):
                self.piano_btn.setEnabled(True)
        
        self.piano.finished.connect(on_close)
        self.piano.show()
        if hasattr(self, 'piano_btn'):
            self.piano_btn.setEnabled(False)

    def playPianoNote(self, note_value):
        """Play note and update frequency fields"""
        try:
            # First update the frequency fields
            self.notesHarmonics(note_value)
            
            # Then play the sound
            octave = self.octave_spinbox.value()
            midi_note = note_value + (octave * 12)
            frequency = 440 * (2 ** ((midi_note - 69) / 12))
            
            duration = 0.5  # seconds
            samples = int(duration * self.fs)
            fade_samples = int(0.02 * self.fs)
            
            t = np.linspace(0, duration, samples, False)
            signal = (0.6 * np.sin(2 * np.pi * frequency * t) +
                     0.3 * np.sin(2 * np.pi * 2 * frequency * t) +
                     0.1 * np.sin(2 * np.pi * 3 * frequency * t))
            
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples) ** 2
                fade_out = np.linspace(1, 0, fade_samples) ** 2
                signal[:fade_samples] *= fade_in
                signal[-fade_samples:] *= fade_out
            
            with sd.OutputStream(samplerate=self.fs, blocksize=2048, channels=1) as stream:
                stream.write(signal.astype(np.float32))
                
        except Exception as e:
            print(f"Error playing note: {e}")

    def update_piano_labels(self):
        """Update all piano key labels based on current octave"""
        current_octave = self.octave_spinbox.value()
        for btn in self.piano_keys:
            note_value = btn.note_value
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_name = note_names[note_value % 12]
            octave = current_octave + (note_value // 12)
            btn.setText(f"{note_name}\n{octave}")



    def decrease_octave(self):
        """Decrease octave and update labels"""
        current = self.octave_spinbox.value()
        if current > 1:
            self.octave_spinbox.setValue(current - 1)

    def increase_octave(self):
        """Increase octave and update labels"""
        current = self.octave_spinbox.value()
        if current < 8:
            self.octave_spinbox.setValue(current + 1)



    def get_note_name(self, note_value):
        """Convert MIDI note number to scientific pitch notation (e.g., C4)"""
        octave = (note_value // 12) + self.octave_spinbox.value()
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return f"{note_names[note_value % 12]}{octave}"

    def _audio_worker(self):
        """Background thread for smooth audio playback"""
        while True:
            signal, fs = self.audio_queue.get()
            if signal is None:  # Exit signal
                break
            try:
                sd.play(signal, fs, blocking=True)
                sd.wait()  # Wait for playback to finish
            except Exception as e:
                print(f"Audio playback error: {e}")

    def playNote(self, note_value):
        """Play note and update frequency fields"""
        try:
            # Calculate frequency
            octave = self.octave_spinbox.value()
            midi_note = note_value + (octave * 12)
            frequency = 440 * (2 ** ((midi_note - 69) / 12))
            
            # Update the frequency fields with harmonics
            self.notesHarmonics(note_value)
            
            # Audio playback (same as before)
            duration = 0.5  # seconds
            samples = int(duration * self.fs)
            fade_samples = int(0.02 * self.fs)
            
            t = np.linspace(0, duration, samples, False)
            signal = (0.6 * np.sin(2 * np.pi * frequency * t) +
                     0.3 * np.sin(2 * np.pi * 2 * frequency * t) +
                     0.1 * np.sin(2 * np.pi * 3 * frequency * t))
            
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples) ** 2
                fade_out = np.linspace(1, 0, fade_samples) ** 2
                signal[:fade_samples] *= fade_in
                signal[-fade_samples:] *= fade_out
            
            with sd.OutputStream(samplerate=self.fs, blocksize=2048, channels=1) as stream:
                stream.write(signal.astype(np.float32))
                
        except Exception as e:
            print(f"Audio error: {e}")

    def notesHarmonics(self, note_value):
        """Update all frequency fields with fundamental and harmonics"""
        try:
            octave = self.octave_spinbox.value()
            # Convert to frequency (A4 = 440Hz is note 69 in MIDI)
            midi_note = note_value + (octave * 12)
            fundfreq = 440 * (2 ** ((midi_note - 69) / 12))
            
            # Set fundamental frequency (1st harmonic)
            self.freq_spinboxes[0].setValue(int(round(fundfreq)))
            self.amp_sliders[0].setValue(100)  # 1.0 amplitude
            
            # Set harmonics (2nd through 6th)
            harmonics = [
                (2, 0.83),  # 2nd harmonic
                (3, 0.67),   # 3rd harmonic
                (4, 0.5),    # 4th harmonic
                (5, 0.33),   # 5th harmonic
                (6, 0.17)    # 6th harmonic
            ]
            
            for i, (multiple, amp) in enumerate(harmonics, start=1):
                if i < len(self.freq_spinboxes):  # Safety check
                    freq = fundfreq * multiple
                    self.freq_spinboxes[i].setValue(int(round(freq)))
                    self.amp_sliders[i].setValue(int(amp * 100))
            
            # Automatically update the plot
            self.plotFAPT()
            
        except Exception as e:
            print(f"Error updating harmonics: {e}")

    def togglePiano(self):
        """Properly manage piano window lifecycle"""
        if not self.pianoOpen:
            self.showPiano()
        else:
            self.piano.close()  # This triggers our close handler

    def plotFAPT(self):
        duration = self.getDuration()
        samples = int(duration * self.fs)
        freqs = self.getFrequencies()
        amps = self.getAmplitudes()
        
        time = np.linspace(0, duration, samples, endpoint=False)
        signal = np.zeros(samples)
        
        for freq, amp in zip(freqs, amps):
            signal += amp * np.sin(2 * np.pi * freq * time)
        
        # Store the full audio signal
        self.full_audio = signal
        self.audio_duration = duration

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
        """Add a 'Load to Controller' button to the matplotlib figure"""
        # Remove previous button if exists
        if hasattr(self, 'load_btn_ax'):
            self.load_btn_ax.remove()
            if hasattr(self, 'span'):
                del self.span
        
        # Store the current audio
        self.full_audio = audio
        self.audio_duration = duration
        
        # Create new button
        self.load_btn_ax = self.fig.add_axes([0.8, 0.01, 0.15, 0.05])  # Position and size
        self.load_btn = Button(self.load_btn_ax, 'Load to Controller')
        
        # Connect the button to our load_to_controller method
        self.load_btn.on_clicked(lambda event: self.load_to_controller())
        
        # Span selector for audio playback and selection
        time = np.linspace(0, duration, len(audio), endpoint=False)
        
        def onselect(xmin, xmax):
            if not hasattr(self, 'full_audio') or len(self.full_audio) <= 1:
                return
                
            ini, end = np.searchsorted(time, (xmin, xmax))
            selected_audio = self.full_audio[ini:end+1].copy()
            
            # Store the selected span for the title
            self.selected_span = (xmin, xmax)
            
            # Apply fade to prevent clicks
            fade_samples = min(int(0.02 * self.fs), len(selected_audio)//4)
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples) ** 2
                fade_out = np.linspace(1, 0, fade_samples) ** 2
                selected_audio[:fade_samples] *= fade_in
                selected_audio[-fade_samples:] *= fade_out
            
            self.selectedAudio = selected_audio
            
            # Play with proper stream management
            try:
                sd.stop()
                sd.play(selected_audio, self.fs, blocking=False)
            except Exception as e:
                print(f"Playback error: {e}")
        
        self.span = SpanSelector(
            self.ax,
            onselect,
            'horizontal',
            useblit=True,
            interactive=True,
            drag_from_anywhere=True
        )
        
        self.canvas.draw()
            
    def load_to_controller(self):
        """Load the current audio to controller (standalone method)"""
        try:
            # First ensure we have audio data
            if not hasattr(self, 'full_audio'):
                self.plotFAPT()  # Regenerate the audio if needed
            
            # Determine which audio to load (selected or full)
            if hasattr(self, 'selectedAudio') and len(self.selectedAudio) > 1:
                audio_to_load = self.selectedAudio
                duration = len(self.selectedAudio) / self.fs
                # Get the time span for the title if available
                if hasattr(self, 'selected_span'):
                    start_time, end_time = self.selected_span
                    title = f"Free Addition {self.format_timestamp(start_time)}-{self.format_timestamp(end_time)}"
                else:
                    title = "Free Addition (selection)"
            else:
                audio_to_load = self.full_audio
                duration = self.audio_duration
                title = "Free Addition"
            
            # Create a minimal controller if needed
            if not hasattr(self.controller, 'adse'):
                from PyQt5.QtWidgets import QWidget
                self.controller = QWidget()
                self.controller.adse = type('', (), {})()
                self.controller.adse.advancedSettings = lambda: print("Advanced settings not available")
            
            # Create new control window
            control_window = ControlMenu(title, self.fs, audio_to_load, duration, self.controller)
            
            # Store reference to the control window
            if not hasattr(self, 'control_windows'):
                self.control_windows = []
            self.control_windows.append(control_window)
            
            # Cleanup handler
            control_window.destroyed.connect(
                lambda: self.control_windows.remove(control_window) 
                if control_window in self.control_windows else None
            )
            
            control_window.show()
            control_window.activateWindow()
        
        except Exception as e:
            print(f"Error loading to controller: {e}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Could not load to controller: {str(e)}")

    # Add this helper method to your class if not already present
    def format_timestamp(self, seconds):
        """Format seconds into MM:SS.mmm format"""
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes):02d}:{seconds:06.3f}"

    def closeEvent(self, event):
        """Clean up when closing"""
        self.audio_queue.put((None, None))  # Stop audio thread
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FreeAdditionPureTones(None)
    window.show()
    sys.exit(app.exec_())