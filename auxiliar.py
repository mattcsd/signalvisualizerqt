from PyQt5.QtWidgets import QDesktopWidget

class Auxiliar:
    def windowGeometry(self, window, width, height, center=False):
        """
        Set the geometry of the window.
        :param window: The window to set the geometry for.
        :param width: The width of the window.
        :param height: The height of the window.
        :param center: If True, center the window on the screen.
        """
        if center:
            # Get the screen dimensions
            screen = QDesktopWidget().screenGeometry()
            screen_width = screen.width()
            screen_height = screen.height()

            # Calculate the x and y coordinates to center the window
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2

            # Set the window geometry
            window.setGeometry(x, y, width, height)
        else:
            # Set the window size without centering
            window.resize(width, height)