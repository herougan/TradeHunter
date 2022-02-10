from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
                             QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
                             QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget, QMessageBox)


def main():
    app = QApplication([])

    window = QWidget()
    layout = QVBoxLayout()

    def on_button_clicked():
        alert = QMessageBox()
        alert.setText('You clicked the button!')
        alert.exec()

    button = QPushButton('Click')
    button.clicked.connect(on_button_clicked)
    layout.addWidget(button)

    layout.addWidget(QPushButton('Top'))
    layout.addWidget(QPushButton('Bottom'))

    window.setLayout(layout)
    window.show()

    app.exec()
    print("Do nothing")


def data_management_window():
    # Load datasets

    # Display datasets in list

    ### UI ###

    # List window (of datasets)

    pass

    def choose_dataset():
        pass

    def modify_dataset():
        # List of valid intervals
        # List of suggested periods

        # Instruments window (Can modify Symbol | Interval | Period)

        # Add instrument

        # Add data (one .csv file)

        # Save
        pass

    def create_dataset():
        pass

    ### Sub-UI ###

    def new_instrument():
        pass

        # Select instrument

        # Instrument custom text

    def modify_instrument():
        pass


def bot_management_window():
    pass


class trade_hunter_app:
    def __init__(self):
        pass

################ Purpose ################
#
# Create and Modify data set definitions - each containing lists of instruments, intervals and periods (tuple) to
# measure.
# Then, data can be downloaded according to the data set definitions. Finally, robots can be evaluated against
# these set definitions (if they are downloaded) and the results will be saved.
# These results can be viewed and graphics can be loaded for more information.
