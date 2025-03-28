import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from tkinter import ttk
from matplotlib.widgets import SpanSelector, Button

from auxiliar import Auxiliar
from controlMenu import ControlMenu

import sys

if sys.platform == "win32":
    from ctypes import windll

    # To avoid blurry fonts
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
else:
    windll = None  # Or handle it differently for macOS
    
class FreeAdditionPureTones(tk.Frame):
    def __init__(self, master, controller):
        tk.Frame.__init__(self, master)
        self.controller = controller
        self.master = master
        self.fs = 48000 # sample frequency
        self.aux = Auxiliar()
        self.cm = ControlMenu()
        self.fig, self.ax = plt.subplots()
        self.selectedAudio = np.empty(1)
        self.pianoOpen = False
        self.freeAddMenu()

    def freeAddMenu(self):
        fam = tk.Toplevel()
        fam.resizable(True, True)
        fam.title('Free addition of pure tones')
        fam.iconbitmap('icons/icon.ico')
        fam.lift() # Place the toplevel window at the top
        # self.aux.windowGeometry(fam, 800, 600)

        # Adapt the window to different sizes
        for i in range(6):
            fam.columnconfigure(i, weight=1)

        for i in range(4):
            fam.rowconfigure(i, weight=1)

        # If the 'generate' menu is closed, close also the generated figure
        def on_closing():
            fam.destroy()
            # if the piano window is opened, close it
            if self.pianoOpen:
                self.piano.destroy()
            plt.close(self.fig)
        fam.protocol("WM_DELETE_WINDOW", on_closing)

        # Read the default values of the atributes from a csv file
        list = self.aux.readFromCsv()
        duration = list[4][2]
        octave = list[4][4]
        freq1, freq2, freq3, freq4, freq5, freq6 = list[4][6], list[4][8], list[4][10], list[4][12], list[4][14], list[4][16]
        ampl1, ampl2, ampl3, ampl4, ampl5, ampl6 = list[4][18], list[4][20], list[4][22], list[4][24], list[4][26], list[4][28]

        # SCALES
        fam.var_amp1 = tk.DoubleVar(value=ampl1)
        fam.var_amp2 = tk.DoubleVar(value=ampl2)
        fam.var_amp3 = tk.DoubleVar(value=ampl3)
        fam.var_amp4 = tk.DoubleVar(value=ampl4)
        fam.var_amp5 = tk.DoubleVar(value=ampl5)
        fam.var_amp6 = tk.DoubleVar(value=ampl6)
        fam.var_dura = tk.DoubleVar(value=duration)

        sca_amp1 = tk.Scale(fam, from_=1, to=0, variable=fam.var_amp1, length=300, orient='vertical', resolution=0.01)
        sca_amp2 = tk.Scale(fam, from_=1, to=0, variable=fam.var_amp2, length=300, orient='vertical', resolution=0.01)
        sca_amp3 = tk.Scale(fam, from_=1, to=0, variable=fam.var_amp3, length=300, orient='vertical', resolution=0.01)
        sca_amp4 = tk.Scale(fam, from_=1, to=0, variable=fam.var_amp4, length=300, orient='vertical', resolution=0.01)
        sca_amp5 = tk.Scale(fam, from_=1, to=0, variable=fam.var_amp5, length=300, orient='vertical', resolution=0.01)
        sca_amp6 = tk.Scale(fam, from_=1, to=0, variable=fam.var_amp6, length=300, orient='vertical', resolution=0.01)
        sca_dura = tk.Scale(fam, from_=1, to=30, variable=fam.var_dura, length=500, orient='horizontal')

        sca_amp1.grid(column=1, row=2, sticky=tk.EW, padx=5, pady=5)
        sca_amp2.grid(column=2, row=2, sticky=tk.EW, padx=5, pady=5)
        sca_amp3.grid(column=3, row=2, sticky=tk.EW, padx=5, pady=5)
        sca_amp4.grid(column=4, row=2, sticky=tk.EW, padx=5, pady=5)
        sca_amp5.grid(column=5, row=2, sticky=tk.EW, padx=5, pady=5)
        sca_amp6.grid(column=6, row=2, sticky=tk.EW, padx=5, pady=5)
        sca_dura.grid(column=1, row=3, sticky=tk.EW, padx=5, pady=5, columnspan=5)

        # ENTRY/SPINBOX
        fam.var_frq1 = tk.DoubleVar(value=freq1)
        fam.var_frq2 = tk.DoubleVar(value=freq2)
        fam.var_frq3 = tk.DoubleVar(value=freq3)
        fam.var_frq4 = tk.DoubleVar(value=freq4)
        fam.var_frq5 = tk.DoubleVar(value=freq5)
        fam.var_frq6 = tk.DoubleVar(value=freq6)
        fam.var_octv = tk.IntVar(value=octave)

        vcmd = (fam.register(self.aux.onValidate), '%S', '%s', '%d')

        ent_frq1 = ttk.Spinbox(fam, from_=0, to=20000, textvariable=fam.var_frq1, validate='key', width=10)
        ent_frq2 = ttk.Spinbox(fam, from_=0, to=20000, textvariable=fam.var_frq2, validate='key', width=10)
        ent_frq3 = ttk.Spinbox(fam, from_=0, to=20000, textvariable=fam.var_frq3, validate='key', width=10)
        ent_frq4 = ttk.Spinbox(fam, from_=0, to=20000, textvariable=fam.var_frq4, validate='key', width=10)
        ent_frq5 = ttk.Spinbox(fam, from_=0, to=24000, textvariable=fam.var_frq5, validate='key', width=10)
        ent_frq6 = ttk.Spinbox(fam, from_=0, to=24000, textvariable=fam.var_frq6, validate='key', width=10)
        ent_dura = ttk.Entry(fam, textvariable=fam.var_dura, validate='key', width=10, validatecommand=vcmd)
        ent_octv = ttk.Spinbox(fam, from_=1, to=6, textvariable=fam.var_octv, validate='key', width=10, state='readonly')
        ent_octv.config(state='disabled')

        ent_frq1.grid(column=1, row=1, sticky=tk.EW, padx=5, pady=5)
        ent_frq2.grid(column=2, row=1, sticky=tk.EW, padx=5, pady=5)
        ent_frq3.grid(column=3, row=1, sticky=tk.EW, padx=5, pady=5)
        ent_frq4.grid(column=4, row=1, sticky=tk.EW, padx=5, pady=5)
        ent_frq5.grid(column=5, row=1, sticky=tk.EW, padx=5, pady=5)
        ent_frq6.grid(column=6, row=1, sticky=tk.EW, padx=5, pady=5)
        ent_dura.grid(column=6, row=3, sticky=tk.S, padx=5, pady=5)
        ent_octv.grid(column=1, row=4, sticky=tk.EW, padx=5, pady=5)

        # LABELS
        lab_ton1 = tk.Label(fam, text='1')
        lab_ton2 = tk.Label(fam, text='2')
        lab_ton3 = tk.Label(fam, text='3')
        lab_ton4 = tk.Label(fam, text='4')
        lab_ton5 = tk.Label(fam, text='5')
        lab_ton6 = tk.Label(fam, text='6')
        lab_freq = tk.Label(fam, text='Frequency (Hz)')
        lab_ampl = tk.Label(fam, text='Amplitude')
        lab_dura = tk.Label(fam, text='Total duration (s)')
        lab_octv = tk.Label(fam, text='Octave')

        lab_ton1.grid(column=1, row=0, sticky=tk.EW)
        lab_ton2.grid(column=2, row=0, sticky=tk.EW)
        lab_ton3.grid(column=3, row=0, sticky=tk.EW)
        lab_ton4.grid(column=4, row=0, sticky=tk.EW)
        lab_ton5.grid(column=5, row=0, sticky=tk.EW)
        lab_ton6.grid(column=6, row=0, sticky=tk.EW)
        lab_freq.grid(column=0, row=1, sticky=tk.E)
        lab_ampl.grid(column=0, row=2, sticky=tk.E)
        lab_dura.grid(column=0, row=3, sticky=tk.S, pady=5)
        lab_octv.grid(column=0, row=4, sticky=tk.E)
        
        # BUTTONS
        but_gene = ttk.Button(fam, command=lambda: self.plotFAPT(fam), text='Plot')
        but_pian = ttk.Button(fam, command=lambda: self.pianoKeyboard(fam, but_pian, ent_octv), text='Show piano')
        but_save = ttk.Button(fam, command=lambda: self.saveDefaultValues(fam, list), text='Save')
        but_help = ttk.Button(fam, command=lambda: self.controller.help.createHelpMenu(2), text='🛈', width=2)

        but_gene.grid(column=6, row=5, sticky=tk.EW, padx=5, pady=5)
        but_pian.grid(column=2, row=4, sticky=tk.EW, padx=5, pady=5)
        but_save.grid(column=6, row=4, sticky=tk.EW, padx=5, pady=5)
        but_help.grid(column=5, row=5, sticky=tk.E, padx=5, pady=5)

        self.plotFAPT(fam)

    def getFrequencies(self, fam):
        frq1 = fam.var_frq1.get()
        frq2 = fam.var_frq2.get()
        frq3 = fam.var_frq3.get()
        frq4 = fam.var_frq4.get()
        frq5 = fam.var_frq5.get()
        frq6 = fam.var_frq6.get()
        return frq1, frq2, frq3, frq4, frq5, frq6
    
    def getAmplitudes(self, fam):
        amp1 = fam.var_amp1.get()
        amp2 = fam.var_amp2.get()
        amp3 = fam.var_amp3.get()
        amp4 = fam.var_amp4.get()
        amp5 = fam.var_amp5.get()
        amp6 = fam.var_amp6.get()
        return amp1, amp2, amp3, amp4, amp5, amp6

    def saveDefaultValues(self, fam, list):
        duration = fam.var_dura.get()
        octave = fam.var_octv.get()
        frq1, frq2, frq3, frq4, frq5, frq6 = self.getFrequencies(fam)
        amp1, amp2, amp3, amp4, amp5, amp6 = self.getAmplitudes(fam)

        new_list = [['NOISE','\t duration', list[0][2],'\t amplitude', list[0][4],'\t fs', list[0][6],'\t noise type', list[0][8]],
                ['PURE TONE','\t duration', list[1][2],'\t amplitude', list[1][4],'\t fs', list[1][6],'\t offset', list[1][8],'\t frequency', list[1][10],'\t phase',  list[1][12]],
                ['SQUARE WAVE','\t duration', list[2][2],'\t amplitude', list[2][4],'\t fs', list[2][6],'\t offset', list[2][8],'\t frequency', list[2][10],'\t phase', list[2][12],'\t active cycle', list[2][14]],
                ['SAWTOOTH WAVE','\t duration', list[3][2],'\t amplitude', list[3][4],'\t fs', list[3][6],'\t offset', list[3][8],'\t frequency', list[3][10],'\t phase', list[3][12],'\t max position', list[3][14]],
                ['FREE ADD OF PT','\t duration', duration,'\t octave', octave,'\t freq1', frq1,'\t freq2', frq2,'\t freq3', frq3,'\t freq4', frq4,'\t freq5', frq5,'\t freq6', frq6,'\t amp1', amp1,'\t amp2', amp2,'\t amp3', amp3,'\t amp4', amp4,'\t amp5', amp5,'\t amp6', amp6],
                ['SPECTROGRAM','\t colormap', list[5][2]]]
        self.aux.saveDefaultAsCsv(new_list)

    def notesHarmonics(self, fam, note):
        # Calculate fundamental frequency of the note
        oct = fam.var_octv.get()
        fundfreq = 440*np.exp(((oct-4)+(note-10)/12)*np.log(2))

        # Configure the fundamental frequency in the slider
        fam.var_frq1.set(value=round(fundfreq,2))
        fam.var_amp1.set(value=1)

        # 2nd harmonic
        freq = fundfreq*2
        fam.var_frq2.set(value=round(freq,2))
        fam.var_amp2.set(value=0.83)

        # 3rd harmonic
        freq = fundfreq*3
        fam.var_frq3.set(value=round(freq,2))
        fam.var_amp3.set(value=0.67)

        # 4th harmonic
        freq = fundfreq*4
        fam.var_frq4.set(value=round(freq,2))
        fam.var_amp4.set(value=0.5)

        # 5th harmonic
        freq = fundfreq*5
        fam.var_frq5.set(value=round(freq,2))
        fam.var_amp5.set(value=0.33)

        # 6th harmonic
        freq = fundfreq*6
        fam.var_frq6.set(value=round(freq,2))
        fam.var_amp6.set(value=0.17)

    def pianoKeyboard(self, fam, but_pian, ent_octv):
        self.pianoOpen = True
        self.piano = tk.Toplevel()
        self.piano.title("Piano")
        self.piano.iconbitmap('icons/icon.ico')
        # self.piano.geometry('{}x200'.format(300))
        but_pian.configure(state='disabled')
        ent_octv.config(state='active')

        # If the piano window is closed, reactivate the "show piano" button
        def on_closing():
            self.pianoOpen = False
            self.piano.destroy()
            but_pian.configure(state='active')
            ent_octv.config(state='disabled')
        self.piano.protocol("WM_DELETE_WINDOW", on_closing)

        white_keys = 7
        black = [1, 1, 0, 1, 1, 1, 0]
        white_notes = [1, 3, 5, 6, 8, 10, 12]
        black_notes = [2, 4, 0, 7, 9, 11]

        for i in range(white_keys):
            btn_white = tk.Button(self.piano, bg='white', activebackground='gray87', command=lambda i=i: self.notesHarmonics(fam, white_notes[i]))
            btn_white.grid(row=0, column=i*3, rowspan=2, columnspan=3, sticky='nsew')

        for i in range(white_keys - 1):
            if black[i]:
                btn_black = tk.Button(self.piano, bg='black', activebackground='gray12', command=lambda i=i: self.notesHarmonics(fam, black_notes[i]))
                btn_black.grid(row=0, column=(i*3)+2, rowspan=1, columnspan=2, sticky='nsew')

        for i in range(white_keys*3):
            self.piano.columnconfigure(i, weight=1)

        for i in range(2):
            self.piano.rowconfigure(i, weight=1)

        # Position the piano window in the middle of the screen
        self.aux.windowGeometry(self.piano, 300, 200, True)

    def plotFAPT(self, fam):
        duration = fam.var_dura.get()
        samples = int(duration*self.fs)
        frq1, frq2, frq3, frq4, frq5, frq6 = self.getFrequencies(fam)
        amp1, amp2, amp3, amp4, amp5, amp6 = self.getAmplitudes(fam)

        time = np.linspace(start=0, stop=duration, num=samples, endpoint=False)
        fapt1 = amp1 * (np.sin(2*np.pi * frq1*time))
        fapt2 = amp2 * (np.sin(2*np.pi * frq2*time))
        fapt3 = amp3 * (np.sin(2*np.pi * frq3*time))
        fapt4 = amp4 * (np.sin(2*np.pi * frq4*time))
        fapt5 = amp5 * (np.sin(2*np.pi * frq5*time))
        fapt6 = amp6 * (np.sin(2*np.pi * frq6*time))
        fapt = fapt1+fapt2+fapt3+fapt4+fapt5+fapt6

        # If the window has been closed, create it again
        if plt.fignum_exists(self.fig.number):
            self.ax.clear() # delete the previous plot
        else:
            self.fig, self.ax = plt.subplots() # create the window

        fig, ax = self.fig, self.ax
        self.addLoadButton(fig, ax, self.fs, time, fapt, duration, fam, 'Free addition of pure tones')
        
        # Plot free addition of pure tones
        limite = max(abs(fapt))*1.1
        ax.plot(time, fapt)
        fig.canvas.manager.set_window_title('Free addition of pure tones')
        ax.set(xlim=[0, duration], ylim=[-limite, limite], xlabel='Time (s)', ylabel='Amplitude')
        ax.axhline(y=0, color='black', linewidth='0.5', linestyle='--') # draw an horizontal line in y=0.0
        ax.grid() # add grid lines

        plt.show()
    
    def addLoadButton(self, fig, ax, fs, time, audio, duration, menu, name):
        # Takes the selected fragment and opens the control menu when clicked
        def load(event):
            if self.selectedAudio.shape == (1,): 
                self.cm.createControlMenu(name, fs, audio, duration, self.controller)
            else:
                time = np.arange(0, len(self.selectedAudio)/fs, 1/fs) # time array of the audio
                durSelec = max(time) # duration of the selected fragment
                self.cm.createControlMenu(name, fs, self.selectedAudio, durSelec, self.controller)
            plt.close(fig)
            menu.destroy()
            axload._but_load = but_load # reference to the Button (otherwise the button does nothing)

        # Adds a 'Load' button to the figure
        axload = fig.add_axes([0.8, 0.01, 0.09, 0.05]) # [left, bottom, width, height]
        but_load = Button(axload, 'Load')
        but_load.on_clicked(load)
        axload._but_load = but_load # reference to the Button (otherwise the button does nothing)

        def listenFrag(xmin, xmax):
            ini, end = np.searchsorted(time, (xmin, xmax))
            self.selectedAudio = audio[ini:end+1]
            sd.play(self.selectedAudio, fs)
            
        self.span = SpanSelector(ax, listenFrag, 'horizontal', useblit=True, interactive=True, drag_from_anywhere=True)
