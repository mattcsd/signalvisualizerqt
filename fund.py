import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import butter, lfilter
import sounddevice as sd
from matplotlib.widgets import Button

# Generate or load a signal
fs = 44100  # Sample rate
duration = 2.0  # Seconds
t = np.linspace(0, duration, int(fs * duration), False)

# Example: A simple harmonic signal (220 Hz fundamental + harmonics)
f0 = 220  # Fundamental frequency
signal = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t) + 0.3 * np.sin(2 * np.pi * 3 * f0 * t)

# Normalize
signal = signal / np.max(np.abs(signal))

# Function to extract fundamental using a bandpass filter
def extract_fundamental(signal, fs, f0, bandwidth=10):
    nyquist = 0.5 * fs
    low = (f0 - bandwidth/2) / nyquist
    high = (f0 + bandwidth/2) / nyquist
    b, a = butter(4, [low, high], btype='band')
    fundamental = lfilter(b, a, signal)
    return fundamental

# Function to remove fundamental using a notch filter
def remove_fundamental(signal, fs, f0, bandwidth=10):
    nyquist = 0.5 * fs
    low = (f0 - bandwidth/2) / nyquist
    high = (f0 + bandwidth/2) / nyquist
    b, a = butter(4, [low, high], btype='bandstop')
    harmonics_only = lfilter(b, a, signal)
    return harmonics_only

# Extract components
fundamental = extract_fundamental(signal, fs, f0)
harmonics_only = remove_fundamental(signal, fs, f0)

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle('Fundamental and Harmonics Separation')

ax1.plot(t, signal, label='Original Signal', color='blue')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.legend()

ax2.plot(t, fundamental, label='Fundamental Only', color='green')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude')
ax2.legend()

ax3.plot(t, harmonics_only, label='Harmonics Only', color='red')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Amplitude')
ax3.legend()

plt.tight_layout()

# Audio playback functions
current_stream = None

def play_signal(signal, fs):
    global current_stream
    if current_stream is not None:
        current_stream.stop()
    current_stream = sd.play(signal, fs)

def stop_playback():
    global current_stream
    if current_stream is not None:
        current_stream.stop()
        current_stream = None

# Add buttons
ax_button_original = plt.axes([0.15, 0.02, 0.2, 0.05])
ax_button_fundamental = plt.axes([0.4, 0.02, 0.2, 0.05])
ax_button_harmonics = plt.axes([0.65, 0.02, 0.2, 0.05])
ax_button_stop = plt.axes([0.4, 0.1, 0.2, 0.05])

button_original = Button(ax_button_original, 'Play Original')
button_fundamental = Button(ax_button_fundamental, 'Play Fundamental')
button_harmonics = Button(ax_button_harmonics, 'Play Harmonics')
button_stop = Button(ax_button_stop, 'Stop Playback')

button_original.on_clicked(lambda event: play_signal(signal, fs))
button_fundamental.on_clicked(lambda event: play_signal(fundamental, fs))
button_harmonics.on_clicked(lambda event: play_signal(harmonics_only, fs))
button_stop.on_clicked(lambda event: stop_playback())

plt.show()