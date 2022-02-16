from PyQt5 import QtCore
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
                             QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
                             QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget, QMessageBox, QBoxLayout, QListWidget, QMainWindow, QListWidgetItem)

# Settings
import settings
from util.dataRetrievalUtil import load_trade_advisor_list, get_dataset_changes, update_specific_dataset_change, \
    write_new_empty_dataset
from util.langUtil import normify_name


class MainApp:
    def __init__(self):
        # self.main_window = 0
        # self.data_management_window = 1
        # self.trade_advisor_window = 2
        # self.data_do_something_window = 11
        # self.bot_detail_window = 21
        # self.bot_testing_suite = 22

        self.save = False
        self.curr_window = -1
        self.options = -1

        self.app = QApplication([])
        pass

    def __open__(self):
        self.curr_window = 0
        self.main_window()

    def main_window(self):
        main_window = QWidget()
        layout = QVBoxLayout()

        dm_button = QPushButton('Data Management')
        ta_button = QPushButton('Trade Advisor')

        def on_dm_click():
            self.window = self.data_management_window()
            self.window.show()
            main_window.close()

        def on_ta_click():
            self.window = self.trade_advisor_window()
            self.window.show()
            main_window.close()

        dm_button.clicked.connect(on_dm_click)
        ta_button.clicked.connect(on_ta_click)
        layout.addWidget(dm_button)
        layout.addWidget(ta_button)

        main_window.setLayout(layout)
        main_window.setWindowTitle('Trade Hunter')
        main_window.show()
        self.app.exec()

        return main_window

    def data_management_window(self):

        keyPressed = QtCore.pyqtSignal(QtCore.QEvent)

        # Save variables
        self.save = False

        window = QWidget()
        head_body_tail = QVBoxLayout()
        head = QHBoxLayout()
        body = QHBoxLayout()
        tail = QHBoxLayout()
        head_body_tail.addLayout(head)
        head_body_tail.addLayout(body)
        head_body_tail.addLayout(tail)

        def build_dataset_list():
            pass

        def build_dataset_table():
            pass

        def dataset_pane():
            layout = QVBoxLayout()

            dataset_select = QListWidget()
            dataset_label = QLabel('Dataset')
            for i in range(10):
                item = QListWidgetItem(F'{i}', dataset_select)
            select_layout = QHBoxLayout()
            select_layout.addWidget(dataset_label)
            select_layout.addWidget(dataset_select)
            layout.addLayout(select_layout)

            mid = QHBoxLayout()
            mid_left = QVBoxLayout()
            mid_right = QVBoxLayout()
            mid_mid = QVBoxLayout()
            # Mid left - Labels
            mid_left.addWidget(QLabel('Symbols'))
            mid_left.addWidget(QLabel('Interval'))
            mid_left.addWidget(QLabel('Period'))
            mid_left.addWidget(QLabel(''))
            # Mid mid - list widgets
            symbol_list = QListWidget()
            interval_list = QListWidget()
            period_list = QListWidget()
            mid_mid.addWidget(symbol_list)
            mid_mid.addWidget(interval_list)
            mid_mid.addWidget(period_list)
            # Mid right - text widgets -> values to add
            symbol_text = QTextEdit()
            interval_text = QTextEdit()
            period_text = QTextEdit()
            mid_right.addWidget(symbol_text)
            mid_right.addWidget(interval_text)
            mid_right.addWidget(period_text)


            add_instrument_button = QPushButton('Add')
            cancel_instrument_button = QPushButton('Reset')
            mid_mid.addWidget(add_instrument_button)
            mid_right.addWidget(cancel_instrument_button)
            mid.addLayout(mid_left)
            mid.addLayout(mid_mid)
            mid.addLayout(mid_right)
            layout.addLayout(mid)

            tail = QHBoxLayout()
            create_button = QPushButton('New')
            save_button = QPushButton('Save')
            create_button.clicked.connect(create_button_clicked)
            save_button.clicked.connect(save_button_clicked)
            tail.addWidget(create_button)
            tail.addWidget(save_button)
            layout.addLayout(tail)

            return layout

        def datatable_pane():
            layout = QVBoxLayout()
            layout.addWidget(QLabel('List of Instruments'))
            table = QTableWidget(20, 3)
            layout.addWidget(table)
            return layout

        def create_button_clicked():
            pass

        def save_button_clicked():
            pass

        def back_button_clicked():
            back()

        def back():
            new_window = self.main_window()
            new_window.show()
            window.close()

        left = dataset_pane()
        right = datatable_pane()
        body.addLayout(left)
        body.addLayout(right)
        body.setStretchFactor(left, 0.5)
        body.setStretchFactor(right, 4)

        def create_dataset():
            keyPressed = QtCore.pyqtSignal(QtCore.QEvent)

            window = QWidget()
            layout = QVBoxLayout()

            name_entry = QLineEdit()
            name_entry.setFocus()
            layout.addWidget(name_entry.get)

            select_button = QPushButton()
            cancel_button = QPushButton()

            def on_name_select():
                write_new_empty_dataset(normify_name(name_entry.text()))
                back()

            def on_cancel():
                back()

            def back():
                window.close()

            def back_key(event):
                print(event.key())
                if event.key() == QtCore.Qt.Key_0:
                    back()

            keyPressed.connect(back_key)
            select_button.clicked.connect(on_name_select)
            name_entry.editingFinished.connect(on_name_select)
            cancel_button.clicked.connect(on_cancel)

            window.setLayout(layout)
            window.setWindowTitle('Select a name')
            window.show()

        update_all_button = QPushButton('Update All')
        back_button = QPushButton('Back')
        tail.addWiget(back_button)
        tail.addWidget(update_all_button)

        window.setLayout(head_body_tail)
        window.windowTitle = 'Data Management'

        return window

    def trade_advisor_window(self):
        window = QWidget()
        layout = QVBoxLayout()

        head = QHBoxLayout()
        body = QVBoxLayout()
        bottom_body = QHBoxLayout()
        tail = QHBoxLayout()

        robot_row = QHBoxLayout()
        input_row = QHBoxLayout()
        button_row = QHBoxLayout()

        robot_row.addWidget(QLabel('Robot'))
        robot_input = QTextEdit()
        robot_row.addWidget(robot_input)

        input_row.addWidget(QLabel('Variables'))
        input_input = QTextEdit()
        new_input_button = QPushButton('New')
        input_row.addWidget(input_input)
        input_row.addWidget(new_input_button)

        test_button = QPushButton('Test')
        optimise_button = QPushButton('Optimise')
        button_row.addWidget(test_button)
        button_row.addWidget(optimise_button)

        body.addLayout(robot_row)
        body.addLayout(input_row)
        body.addLayout(button_row)

        bottom_body.addLayout(self.trade_advisor_details_pane(), 2)

        # left = QVBoxLayout()
        # right = QVBoxLayout()
        # layout.addLayout(left)
        # layout.addLayout(right)

        layout.addLayout(head)
        layout.addLayout(body)
        layout.addLayout(bottom_body)
        layout.addLayout(tail)

        window.setWindowTitle('Trade Advisors')
        window.setLayout(layout)

        return window

    def trade_advisor_details_pane(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel('Details'))
        details_layout = QHBoxLayout()
        details_layout.addWidget(QLabel('None'))

        layout.addLayout(details_layout)
        return layout

    def create_dataset_window(self):
        window = QWidget()
        layout = QVBoxLayout()
        pass

    def add_instrument_window(self):
        window = QWidget()
        layout = QVBoxLayout()
        pass

    # Minor windows

    def generic_context_window(self, args, previous_window):
        window = QWidget()
        layout = QVBoxLayout()
        self.options = -1

        for i in range(args):
            def option_click(self):
                self.options = i
                previous_window.show()
                window.destroy()

            button = QPushButton(args[i])
            button.clicked.connect(option_click)
            layout.addWidget(button)

        window.setLayout(layout)
        window.setWindowTitle('Select an Option')
        window.show()

    def select_trade_advisor_window(self):
        window = QWidget()
        layout = QBoxLayout()

        select_list = QListWidget()
        select_list.items(load_trade_advisor_list())
        layout.addWidget(select_list)

        window.setLayout(layout)
        window.setWindowTitle('Select a trade advisor')
        window.show()

    def select_dataset_window(self):
        window = QWidget()
        layout = QBoxLayout()

        select_list = QListWidget()
        select_list.items(load_trade_advisor_list())
        layout.addWidget(select_list)

        window.setLayout(layout)
        window.setWindowTitle('Select a trade advisor')
        window.show()

    def update_all_with_progress_bar(self, title):
        """Redownload data according to updated datasets.
        Returns to data management window."""
        dsc = get_dataset_changes()

        window = QWidget()
        layout = QBoxLayout()

        progress_bar = QProgressBar()
        layout.addWidget(progress_bar)

        window.setLayout(layout)
        window.setWindowTitle(title)
        window.show()

        l = len(dsc['index'])
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(l)
        for i in range(dsc['index']):
            update_specific_dataset_change(dsc['index'][i])
            progress_bar.setValue(i)


def main():
    app = QApplication([])

    window = QWidget()
    layout = QVBoxLayout()

    def on_data_clicked():
        alert = QMessageBox()
        alert.setText('You clicked the button!')

        def on_alert_clicked():
            window.show()

        alert.buttonClicked.connect(on_alert_clicked)
        window.hide()
        alert.exec()

    def on_bot_clicked():
        alert = QMessageBox()
        alert.setText('You clicked the button!')
        alert.exec()
        window.hide()

    data_button = QPushButton('Data Management')
    data_button.clicked.connect(on_data_clicked)
    bot_button = QPushButton('Trade Advisors')
    bot_button.clicked.connect(on_bot_clicked)

    layout.addWidget(data_button)
    layout.addWidget(bot_button)

    # table = QTableWidget(20, 3)
    # layout.addWidget(table)
    window.windowTitle = 'Trade Hunter'

    # title = QLabel('Trade Hunter py')
    # layout.addWidget(title)
    #
    # button = QPushButton('Click')
    # button.clicked.connect(on_button_clicked)
    # layout.addWidget(button)
    #
    # layout.addWidget(QPushButton('Top'))
    # layout.addWidget(QPushButton('Bottom'))
    #
    # table = QTableWidget(20, 3)
    # layout.addWidget(table)
    #
    # e1button = QPushButton('Extra1')
    # e2button = QPushButton('Extra2')
    #
    # sub_layout = QHBoxLayout()
    # sub_layout.addWidget(e1button)
    # sub_layout.addWidget(e2button)
    # layout.addLayout(sub_layout)

    window.setLayout(layout)
    window.show()

    app.exec()

################ Purpose ################
#
# Create and Modify data set definitions - each containing lists of instruments, intervals and periods (tuple) to
# measure.
# Then, data can be downloaded according to the data set definitions. Finally, robots can be evaluated against
# these set definitions (if they are downloaded) and the results will be saved.
# These results can be viewed and graphics can be loaded for more information.
