from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QGridLayout, QLabel, 
                            QPushButton, QRadioButton, QComboBox, QMessageBox,
                            QButtonGroup)
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from auxiliar import Auxiliar

class Spectrogram(QDialog):
    def __init__(self, parent=None, controller=None):
        super().__init__(parent)
        self.controller = controller
        self.aux = Auxiliar()
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle('Choose colormap of the spectrogram')
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        # Initialize colormap data
        self.cmaps = [('Perceptually Uniform Sequential', [
                    'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
                ('Sequential', [
                    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
                ('Sequential (2)', [
                    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                    'hot', 'afmhot', 'gist_heat', 'copper']),
                ('Diverging', [
                    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
                ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
                ('Qualitative', [
                    'Pastel1', 'Pastel2', 'Paired', 'Accent',
                    'Dark2', 'Set1', 'Set2', 'Set3',
                    'tab10', 'tab20', 'tab20b', 'tab20c']),
                ('Miscellaneous', [
                    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                    'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
                    'gist_ncar'])]

        # Load current colormap from CSV with error handling
        self.current_colormap = 'viridis'  # Default fallback
        try:
            csv_data = self.aux.readFromCsv()
            if len(csv_data) > 5 and len(csv_data[5]) > 2:
                self.current_colormap = csv_data[5][2]
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not load colormap preference: {str(e)}")

        # Create UI elements
        self.create_radio_buttons()
        self.create_dropdowns()
        self.create_buttons()
        
        # Set initial state
        self.set_initial_selection()
        
        # Layout setup
        main_layout = QVBoxLayout()
        grid_layout = QGridLayout()
        
        # Add widgets to grid
        grid_layout.addWidget(self.rdb_pusq, 0, 0)
        grid_layout.addWidget(self.dd_pusq, 1, 0)
        grid_layout.addWidget(self.but_pusq, 1, 1)
        
        grid_layout.addWidget(self.rdb_sequ, 2, 0)
        grid_layout.addWidget(self.dd_sequ, 3, 0)
        grid_layout.addWidget(self.but_sequ, 3, 1)
        
        grid_layout.addWidget(self.rdb_seq2, 4, 0)
        grid_layout.addWidget(self.dd_seq2, 5, 0)
        grid_layout.addWidget(self.but_seq2, 5, 1)
        
        grid_layout.addWidget(self.rdb_dive, 6, 0)
        grid_layout.addWidget(self.dd_dive, 7, 0)
        grid_layout.addWidget(self.but_dive, 7, 1)
        
        grid_layout.addWidget(self.rdb_cycl, 8, 0)
        grid_layout.addWidget(self.dd_cycl, 9, 0)
        grid_layout.addWidget(self.but_cycl, 9, 1)
        
        grid_layout.addWidget(self.rdb_qual, 10, 0)
        grid_layout.addWidget(self.dd_qual, 11, 0)
        grid_layout.addWidget(self.but_qual, 11, 1)
        
        grid_layout.addWidget(self.rdb_misc, 12, 0)
        grid_layout.addWidget(self.dd_misc, 13, 0)
        grid_layout.addWidget(self.but_misc, 13, 1)
        
        main_layout.addLayout(grid_layout)
        main_layout.addWidget(self.but_save)
        self.setLayout(main_layout)

    def create_radio_buttons(self):
        self.rdb_pusq = QRadioButton('Perceptually Uniform Sequential')
        self.rdb_sequ = QRadioButton('Sequential')
        self.rdb_seq2 = QRadioButton('Sequential (2)')
        self.rdb_dive = QRadioButton('Diverging')
        self.rdb_cycl = QRadioButton('Cyclic')
        self.rdb_qual = QRadioButton('Qualitative')
        self.rdb_misc = QRadioButton('Miscellaneous')
        
        self.rdb_group = QButtonGroup()
        self.rdb_group.addButton(self.rdb_pusq, 1)
        self.rdb_group.addButton(self.rdb_sequ, 2)
        self.rdb_group.addButton(self.rdb_seq2, 3)
        self.rdb_group.addButton(self.rdb_dive, 4)
        self.rdb_group.addButton(self.rdb_cycl, 5)
        self.rdb_group.addButton(self.rdb_qual, 6)
        self.rdb_group.addButton(self.rdb_misc, 7)
        
        self.rdb_group.buttonClicked.connect(self.update_dropdown_state)

    def create_dropdowns(self):
        self.dd_pusq = QComboBox()
        self.dd_pusq.addItems(self.cmaps[0][1])
        self.dd_sequ = QComboBox()
        self.dd_sequ.addItems(self.cmaps[1][1])
        self.dd_seq2 = QComboBox()
        self.dd_seq2.addItems(self.cmaps[2][1])
        self.dd_dive = QComboBox()
        self.dd_dive.addItems(self.cmaps[3][1])
        self.dd_cycl = QComboBox()
        self.dd_cycl.addItems(self.cmaps[4][1])
        self.dd_qual = QComboBox()
        self.dd_qual.addItems(self.cmaps[5][1])
        self.dd_misc = QComboBox()
        self.dd_misc.addItems(self.cmaps[6][1])

    def create_buttons(self):
        self.but_pusq = QPushButton('Show colormap')
        self.but_pusq.clicked.connect(lambda: self.plot_color_gradients(self.cmaps[0]))
        
        self.but_sequ = QPushButton('Show colormap')
        self.but_sequ.clicked.connect(lambda: self.plot_color_gradients(self.cmaps[1]))
        
        self.but_seq2 = QPushButton('Show colormap')
        self.but_seq2.clicked.connect(lambda: self.plot_color_gradients(self.cmaps[2]))
        
        self.but_dive = QPushButton('Show colormap')
        self.but_dive.clicked.connect(lambda: self.plot_color_gradients(self.cmaps[3]))
        
        self.but_cycl = QPushButton('Show colormap')
        self.but_cycl.clicked.connect(lambda: self.plot_color_gradients(self.cmaps[4]))
        
        self.but_qual = QPushButton('Show colormap')
        self.but_qual.clicked.connect(lambda: self.plot_color_gradients(self.cmaps[5]))
        
        self.but_misc = QPushButton('Show colormap')
        self.but_misc.clicked.connect(lambda: self.plot_color_gradients(self.cmaps[6]))
        
        self.but_save = QPushButton('Save')
        self.but_save.clicked.connect(self.save_colormap)

    def set_initial_selection(self):
        # Find which category contains our current colormap
        for i, (category, cmap_list) in enumerate(self.cmaps):
            if self.current_colormap in cmap_list:
                self.rdb_group.button(i+1).setChecked(True)
                combo = getattr(self, f'dd_{["pusq", "sequ", "seq2", "dive", "cycl", "qual", "misc"][i]}')
                combo.setCurrentText(self.current_colormap)
                self.update_dropdown_state()
                break

    def update_dropdown_state(self):
        selected_id = self.rdb_group.checkedId()
        self.dd_pusq.setEnabled(selected_id == 1)
        self.but_pusq.setEnabled(selected_id == 1)
        self.dd_sequ.setEnabled(selected_id == 2)
        self.but_sequ.setEnabled(selected_id == 2)
        self.dd_seq2.setEnabled(selected_id == 3)
        self.but_seq2.setEnabled(selected_id == 3)
        self.dd_dive.setEnabled(selected_id == 4)
        self.but_dive.setEnabled(selected_id == 4)
        self.dd_cycl.setEnabled(selected_id == 5)
        self.but_cycl.setEnabled(selected_id == 5)
        self.dd_qual.setEnabled(selected_id == 6)
        self.but_qual.setEnabled(selected_id == 6)
        self.dd_misc.setEnabled(selected_id == 7)
        self.but_misc.setEnabled(selected_id == 7)

    def plot_color_gradients(self, cmap_category):
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        
        cmap_list = cmap_category[1]
        nrows = len(cmap_list)
        figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
        fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
        fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)

        axs[0].set_title(f"{cmap_category[0]} colormaps", fontsize=14)

        for ax, cmap_name in zip(axs, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=cmap_name)
            ax.text(-.01, .5, cmap_name, va='center', ha='right', fontsize=10,
                    transform=ax.transAxes)

        for ax in axs:
            ax.set_axis_off()

        plt.show()

    def save_colormap(self):
        try:
            selected_id = self.rdb_group.checkedId()
            if selected_id == 1:
                choice = self.dd_pusq.currentText()
            elif selected_id == 2:
                choice = self.dd_sequ.currentText()
            elif selected_id == 3:
                choice = self.dd_seq2.currentText()
            elif selected_id == 4:
                choice = self.dd_dive.currentText()
            elif selected_id == 5:
                choice = self.dd_cycl.currentText()
            elif selected_id == 6:
                choice = self.dd_qual.currentText()
            elif selected_id == 7:
                choice = self.dd_misc.currentText()
            else:
                choice = 'viridis'  # fallback

            # Read current CSV data
            csv_data = self.aux.readFromCsv()
            
            # Ensure proper data structure
            while len(csv_data) < 6:
                csv_data.append([])
            while len(csv_data[5]) < 3:
                csv_data[5].append("")
                
            csv_data[5][2] = choice
            self.aux.saveDefaultAsCsv(csv_data)
            
            QMessageBox.information(self, "Saved", "Colormap preference saved successfully")
            self.close()
            plt.close('all')
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save colormap: {str(e)}")