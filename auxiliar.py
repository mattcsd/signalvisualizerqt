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