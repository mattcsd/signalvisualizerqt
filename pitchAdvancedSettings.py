# pitch_advanced_settings.py

from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QVBoxLayout

class AdvancedSettings(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Pitch Settings")

        # Add your PyQt5 widgets here (e.g., QLineEdit, QCheckBox, etc.)
        # For now, keep it simple:
        self.settings = {
            "autocorr": {"minf0": 75, "maxf0": 500},
            "subharmonics": {"n_candidates": 5},
            "spinet": {"use_spinet": False},
            "other": {"some_other_setting": 1}
        }

        self.accept()  # Automatically close the dialog for now (just for testing)

    def getAutocorrelationVars(self):
        return self.settings["autocorr"]

    def getSubharmonicsVars(self):
        return self.settings["subharmonics"]

    def getSpinetVars(self):
        return self.settings["spinet"]

    def getVariables(self):
        return self.settings["other"]


class PitchAdvancedSettingsHandler:
    def __init__(self, parent=None):
        self.parent = parent
        self.settings = {}

    def advancedSettings(self):
        dialog = AdvancedSettings(self.parent)
        if dialog.exec_():  # Only store settings if OK/Accept is clicked
            self.settings['autocorr'] = dialog.getAutocorrelationVars()
            self.settings['subharmonics'] = dialog.getSubharmonicsVars()
            self.settings['spinet'] = dialog.getSpinetVars()
            self.settings['other'] = dialog.getVariables()

            print("Saved pitch settings:", self.settings)
