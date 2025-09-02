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
from config import BASE_DIR, RECORDINGS_DIR, LIBRARY_DIR

MAX_WINDOWS = 5 

class Load(QWidget):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.master = master
        self.selectedAudio = np.empty(1)
        self.fs = 44100  # Default sample rate
        self.file_path = ""

        self.control_windows = []  # List to track all open control windows
        self.selected_span = (0, 0)  # Track selected time span
        
        self.controller = controller  # This should reference your Start instance

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
            str(library_dir),  # Use library_dir instead of LIBRARY_DIR
            "Audio Files (*.wav *.mp3);;WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*)"
        )
        
        if not file_path:  # User cancelled
            return
            
        self.file_path = file_path
        
        try:
            # Read audio file using librosa (supports both WAV and MP3)
            audio, self.fs = librosa.load(file_path, sr=None, mono=False)
            
            # Check if stereo and convert to mono if needed
            if audio.ndim > 1:
                QMessageBox.warning(
                    self, 
                    "Stereo File", 
                    "This file is in stereo mode. It will be converted to mono."
                )
                ampMax = np.max(np.abs(audio))
                audio = np.mean(audio, axis=0)  # Convert to mono by averaging channels
                audio = audio * ampMax / np.max(np.abs(audio))  # Normalize
            
            self.plotAudio(audio)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load file: {str(e)}")
    
    def plotAudio(self, audio):
        self.ax.clear()

        # Reset selected span when loading new audio
        self.selected_span = None
        self.selectedAudio = np.empty(1)
        
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
            self.selected_span = (xmin, xmax)  # Store the selected span
            sd.play(self.selectedAudio, self.fs)
            
        self.span = SpanSelector(
            self.ax,
            on_select,
            'horizontal',
            useblit=True,
            interactive=True,
            drag_from_anywhere=True
        )

    def format_timestamp(self, seconds):
        """Convert seconds to mm:ss format"""
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"[:8]  # Shows mm:ss.xx

    def addLoadButton(self):
        # Remove existing button if it exists
        if hasattr(self, 'load_button_ax'):
            self.fig.delaxes(self.load_button_ax)
            
        # Create button axes
        self.load_button_ax = self.fig.add_axes([0.8, 0.01, 0.15, 0.05])
        self.load_button = Button(self.load_button_ax, 'Load to Controller')
        
        def on_load(event):
            if len(self.control_windows) >= MAX_WINDOWS:
                oldest = self.control_windows.pop(0)
                oldest.close()
            if self.selectedAudio.shape == (1,):  # No selection, use entire audio
                audio_to_load = self.ax.lines[0].get_ydata()
                duration = len(audio_to_load) / self.fs
                start_time = 0
                end_time = duration
            else:
                audio_to_load = self.selectedAudio
                duration = len(audio_to_load) / self.fs
                start_time, end_time = self.selected_span
                
            # Create window title with span
            name = Path(self.file_path).stem
            if self.selectedAudio.shape != (1,):  # Only show span if selection was made
                title = f"{name} {self.format_timestamp(start_time)}-{self.format_timestamp(end_time)}"
            else:
                title = name
                
            # Create new control window
            control_window = ControlMenu(title, self.fs, audio_to_load, duration, self.controller)
            
            if hasattr(self.controller, 'update_windows_menu'):
                self.controller.update_windows_menu()
                
            # Store the title early since windowTitle() may fail later
            window_title = control_window.windowTitle()
            
            def handle_close():
                try:
                    # Check if window still exists in the list
                    if control_window in self.control_windows:
                        self.control_windows.remove(control_window)
                        print(f"Removed window: '{window_title}'. Total windows: {len(self.control_windows)}")
                        # Print all remaining windows
                        print("Current windows:", [w.base_name for w in self.control_windows])

                    else:
                        print(f"Window '{window_title}' not found in control_windows list")
                except RuntimeError:
                    # This catches cases where the window is partially destroyed
                    print(f"Window '{window_title}' already destroyed during cleanup")
                
            control_window.destroyed.connect(handle_close)
            
            self.control_windows.append(control_window)
            print(f"Added window: '{control_window.windowTitle()}'. Total windows: {len(self.control_windows)}")
            print("All windows:", [w.base_name for w in self.control_windows])            
            
            control_window.show()
            
            control_window.activateWindow()

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