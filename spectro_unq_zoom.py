def plot_spectrogram(self):
        if self.audio_data is None:
            return
            
        self.figure.clear()
        self.playback_lines = []
        
        # Create 3 independent subplots
        gs = self.figure.add_gridspec(3, 1, height_ratios=[1, 1, 2], hspace=0.6)
        
        # 1. Waveform plot (independent)
        ax0 = self.figure.add_subplot(gs[0])
        ax0.plot(self.time, self.audio_data, color='b', linewidth=0.5, alpha=0.7)
        self.playback_lines.append(ax0.axvline(x=0, color='r', linewidth=1, animated=True))
        ax0.set_title("Waveform")
        ax0.set_xlim(0, self.time[-1])
        
        # 2. Amplitude envelope plot (independent)
        ax1 = self.figure.add_subplot(gs[1])  # No longer shares x-axis
        amplitude = np.abs(self.audio_data)
        smooth_window = int(0.02 * self.sample_rate)
        amplitude_smooth = np.convolve(amplitude, np.ones(smooth_window)/smooth_window, mode='same')
        ax1.plot(self.time, amplitude_smooth, 'b-', linewidth=1)
        self.playback_lines.append(ax1.axvline(x=0, color='r', linewidth=1, animated=True))
        ax1.set_title("Amplitude Envelope")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, self.time[-1])  # Manually set same limits
        
        # 3. Spectrogram plot (independent)
        ax2 = self.figure.add_subplot(gs[2])  # No longer shares x-axis
        D = librosa.stft(
            self.audio_data,
            n_fft=self.window_size_spin.value(),
            hop_length=self.hop_size_spin.value(),
            win_length=self.window_size_spin.value()
        )
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(
            S_db,
            sr=self.sample_rate,
            hop_length=self.hop_size_spin.value(),
            x_axis='time',
            y_axis='linear',
            ax=ax2,
            cmap='viridis',
            vmin=-60,
            vmax=0
        )
        ax2.set_ylim(0, self.max_freq_spin.value())
        self.playback_lines.append(ax2.axvline(x=0, color='r', linewidth=1, animated=True))
        ax2.set_title("Spectrogram")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlabel("Time (s)")
        ax2.set_xlim(0, self.time[-1])  # Manually set same initial limits
        
        # Draw everything once
        self.canvas.draw()
        
        # Store the background without the cursor lines
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        
        # Mark the cursor lines as animated
        for line in self.playback_lines:
            line.set_animated(True)