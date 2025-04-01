from PyQt5.QtWidgets import QFileDialog, QMessageBox
from scipy.io.wavfile import write
import numpy as np
from matplotlib.widgets import Button
from PyQt5.QtGui import QDoubleValidator, QIntValidator

# Add this to your auxiliar.py file
class Auxiliar:
    def readFromCsv(self):
        # This is a simplified version - you should replace with your actual CSV reading logic
        return [
            ['NOISE', '\t duration', 1.0, '\t amplitude', 0.5, '\t fs', 44100, '\t noise type', 'white'],
            ['PURE TONE', '\t duration', 1.0, '\t amplitude', 0.5, '\t fs', 44100, 
             '\t offset', 0.0, '\t frequency', 440, '\t phase', 0.0],
            # Add other default values as needed
        ]
        
    def get_double_validator(self):
        """Returns a QDoubleValidator for decimal numbers"""
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        return validator
        
    def get_int_validator(self):
        """Returns a QIntValidator for whole numbers"""
        return QIntValidator()

    def saveasWavCsv(self, controller, fig, time, audio, position, fs):
        """Save audio data as both WAV and CSV files"""
        try:
            # Create a button for saving
            ax_save = fig.add_axes([0.7, position, 0.1, 0.05])  # [left, bottom, width, height]
            btn_save = Button(ax_save, 'Save Data')
            
            def save_data(event):
                # Get save path from file dialog
                path, _ = QFileDialog.getSaveFileName(
                    controller,  # Use the controller as parent
                    "Save Audio Data", 
                    "", 
                    "WAV files (*.wav);;CSV files (*.csv);;All Files (*)"
                )
                
                if path:
                    if path.endswith('.wav'):
                        # Save as WAV file
                        write(path, fs, audio)
                    elif path.endswith('.csv'):
                        # Save as CSV file
                        data = np.column_stack((time, audio))
                        np.savetxt(path, data, delimiter=',', 
                                  header='Time,Amplitude', comments='')
                    else:
                        # Default to WAV if no extension
                        write(path + '.wav', fs, audio)
                        
            btn_save.on_clicked(save_data)
            return btn_save
            
        except Exception as e:
            QMessageBox.critical(controller, "Save Error", f"Failed to save data: {str(e)}")
            return None

    def saveasCsv(self, fig, x_data, y_data, position, title=""):
        """Save plot data as CSV file"""
        try:
            ax_save = fig.add_axes([0.7, position, 0.1, 0.05])
            btn_save = Button(ax_save, f'Save {title}')
            
            def save_data(event):
                path, _ = QFileDialog.getSaveFileName(
                    None,  # Can pass controller here if needed
                    f"Save {title} Data",
                    "",
                    "CSV files (*.csv);;All Files (*)"
                )
                
                if path:
                    if not path.endswith('.csv'):
                        path += '.csv'
                    data = np.column_stack((x_data, y_data))
                    np.savetxt(path, data, delimiter=',', 
                              header=f'{title}_X,{title}_Y', comments='')
                    
            btn_save.on_clicked(save_data)
            return btn_save
            
        except Exception as e:
            QMessageBox.critical(None, "Save Error", f"Failed to save CSV: {str(e)}")
            return None


    def saveDefaultAsCsv(self, data):
        # Placeholder for your CSV saving logic
        pass
    
    def bigFrequency(self, frequency, fs):
        # Check if frequency is valid
        if frequency > fs/2:
            QMessageBox.warning(None, "Warning", 
                              f"Frequency {frequency}Hz is greater than Nyquist frequency ({fs/2}Hz)")
    
    def onValidate(self, text, prior_text, action):
        # Basic validation - allow numbers and decimal point
        if text in '0123456789.':
            return True
        return False
    
    def onValidateInt(self, text):
        # Basic integer validation
        if text in '0123456789':
            return True
        return False