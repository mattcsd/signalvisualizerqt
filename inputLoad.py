import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import struct
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector, Button
from matplotlib.figure import Figure
from controlMenu import ControlMenu

class Load(QWidget):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.master = master
        self.selectedAudio = np.empty(1)
        self.fs = 44100  # Default sample rate
        self.file_path = ""
        
        self.setupUI()
        
    def setupUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create open file button
        self.open_button = QPushButton('Open Audio File')
        self.open_button.clicked.connect(self.loadAudio)
        
        # Figure setup
        self.fig = Figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Add widgets to layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_button)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        
        self.setLayout(main_layout)
        
    def loadAudio(self):
        # Get the directory of the main window
        main_window_dir = Path(self.master.window().windowFilePath()).parent if hasattr(self.master, 'window') else Path.cwd()
        library_dir = main_window_dir / "library"
        
        # Create library directory if it doesn't exist
        if not library_dir.exists():
            library_dir.mkdir()
            QMessageBox.information(
                self,
                "Library Directory Created",
                f"The 'library' directory was created at:\n{library_dir}"
            )
        
        # Open file dialog starting in the library directory
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Audio File", 
            str(library_dir),  # Set initial directory to library
            "WAV Files (*.wav);;All Files (*)"
        )
        
        if not file_path:  # User cancelled
            return
            
        self.file_path = file_path
        
        try:
            # Read audio file
            audio, self.fs = sf.read(file_path, dtype='float32')
            
            # Check if stereo and convert to mono if needed
            with open(file_path, 'rb') as wav:
                header_beginning = wav.read(0x18)
                wavChannels, = struct.unpack_from('<H', header_beginning, 0x16)
                if wavChannels > 1:
                    QMessageBox.warning(
                        self, 
                        "Stereo File", 
                        "This file is in stereo mode. It will be converted to mono."
                    )
                    ampMax = np.max(np.abs(audio))
                    audio = np.sum(audio, axis=1)  # Convert to mono
                    audio = audio * ampMax / np.max(np.abs(audio))  # Normalize
            
            self.plotAudio(audio)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load file: {str(e)}")
            
    def plotAudio(self, audio):
        self.ax.clear()
        
        # Calculate time array
        duration = librosa.get_duration(filename=self.file_path)
        time = np.linspace(0, duration, len(audio), endpoint=False)
        
        # Plot the audio
        self.ax.plot(time, audio, linewidth=1)
        self.ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        self.ax.set(
            xlim=[0, duration],
            xlabel='Time (s)',
            ylabel='Amplitude',
            title=Path(self.file_path).stem
        )
        self.ax.grid(True, linestyle=':', alpha=0.5)
        
        # Add load button
        self.addLoadButton()
        
        # Setup span selector for audio selection
        self.setupSpanSelector(time, audio)
        
        self.canvas.draw()
        
    def setupSpanSelector(self, time, audio):
        # Remove existing span selector if it exists
        if hasattr(self, 'span'):
            self.span.disconnect_events()
            del self.span
            
        def on_select(xmin, xmax):
            if len(audio) <= 1:
                return
                
            idx_min = np.argmax(time >= xmin)
            idx_max = np.argmax(time >= xmax)
            self.selectedAudio = audio[idx_min:idx_max]
            sd.play(self.selectedAudio, self.fs)
            
        self.span = SpanSelector(
            self.ax,
            on_select,
            'horizontal',
            useblit=True,
            interactive=True,
            drag_from_anywhere=True
        )
        
    def addLoadButton(self):
        # Remove existing button if it exists
        if hasattr(self, 'load_button_ax'):
            self.fig.delaxes(self.load_button_ax)
            
        # Create button axes
        self.load_button_ax = self.fig.add_axes([0.8, 0.01, 0.15, 0.05])
        self.load_button = Button(self.load_button_ax, 'Load to Controller')
        
        def on_load(event):
            if self.selectedAudio.shape == (1,):  # No selection, use entire audio
                audio_to_load = self.ax.lines[0].get_ydata()
                duration = len(audio_to_load) / self.fs  # Calculate duration from audio length
            else:
                audio_to_load = self.selectedAudio
                duration = len(audio_to_load) / self.fs  # Calculate duration from selected audio
                
            # Create control menu
            name = Path(self.file_path).stem
            

            # Create and show window
            self.control_menu_ref = ControlMenu(name, self.fs, audio_to_load, duration, self.controller)
            
            # Force window to front
            self.control_menu_ref.show()
            #self.control_menu_ref.raise_()
            self.control_menu_ref.activateWindow()

            print("Show called")  # Debug

        self.load_button.on_clicked(on_load)
        
    def showHelp(self):
        QMessageBox.information(
            self, 
            "Help", 
            "Audio File Loader Help\n\n"
            "1. Click 'Open Audio File' to browse for a WAV file\n"
            "2. Select a portion of the audio with your mouse to play just that section\n"
            "3. Click 'Load to Controller' to send the audio to the control menu\n"
            "   - If no selection is made, the entire file will be loaded"
        )