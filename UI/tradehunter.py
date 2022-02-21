from PyQt5 import QtCore
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
                             QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
                             QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget, QMessageBox, QBoxLayout, QListWidget, QMainWindow, QListWidgetItem)

# Settings
import sys
import settings
from UI.QTUtil import get_sheet
from util.dataRetrievalUtil import load_trade_advisor_list, get_dataset_changes, update_specific_dataset_change, \
    write_new_empty_dataset, load_dataset_list, save_dataset, add_as_dataset_change
from util.langUtil import normify_name


class TradeHunterApp():

    def __init__(self):
        self.app = QApplication(sys.argv)
        main = self.MainWindow()
        main.show()
        self.app.exec()

    class MainWindow(QWidget):

        def __init__(self):
            self.keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
            super().__init__()
            self.window()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Space:
                pass
            print(event.key())

        def window(self):
            layout = QVBoxLayout()

            dm_button = QPushButton('Data Management')
            ta_button = QPushButton('Trade Advisor')

            def on_dm_click():
                self.dm_window = TradeHunterApp.DataManagementPage()
                self.dm_window.show()
                self.close()

            def on_ta_click():
                self.ta_window = TradeHunterApp.TradeAdvisorPage()
                self.ta_window.show()
                self.close()

            dm_button.clicked.connect(on_dm_click)
            ta_button.clicked.connect(on_ta_click)
            layout.addWidget(dm_button)
            layout.addWidget(ta_button)

            self.setLayout(layout)
            self.setWindowTitle('Trade Hunter')
            self.show()

    class DataManagementPage(QWidget):
        def __init__(self):
            self.keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
            super().__init__()
            self.window()
            self.windowTitle = 'Data Management'

            self.datasetpane = None
            self.datatablepane = None

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Space:
                pass
            elif event.key() == Qt.Key_Escape:

                pass
            print(event.key())

        def window(self):

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

            def back_button_clicked():
                back()

            def back():
                window = TradeHunterApp.MainWindow()
                window.show()
                self.close()

            # window.keyPressEvent(back)

            left = TradeHunterApp.DataManagementPage.DatasetPane()
            self.datasetpane = left
            right = TradeHunterApp.DataManagementPage.DatatablePane()
            self.datatablepane = right
            left.bind(right.table)

            body.addLayout(left)
            body.addLayout(right)
            body.setStretchFactor(left, 0.5)
            body.setStretchFactor(right, 4)

            update_all_button = QPushButton('Update All')
            back_button = QPushButton('Back')
            back_button.clicked.connect(back)
            tail.addWidget(back_button)
            tail.addWidget(update_all_button)

            self.setLayout(head_body_tail)

        class DatasetPane(QVBoxLayout):
            def __init__(self):
                super().__init__()
                self.table = None
                self.select = None
                self.window()

            def window(self):

                dataset_select = QListWidget()
                dataset_label = QLabel('Dataset')

                self.select = dataset_select
                self.build_dataset_list()

                def load_selected_dataset(event):
                    if event:
                        print("Current item changed", event.text)
                        self.build_dataset_instruments(event.text)
                    else:
                        print("Current item is cringe!")
                        self.build_dataset_instruments(None)
                dataset_select.currentItemChanged.connect(load_selected_dataset)

                select_layout = QHBoxLayout()
                select_layout.addWidget(dataset_label)
                select_layout.addWidget(dataset_select)
                self.addLayout(select_layout)

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
                self.addLayout(mid)

                def create_button_clicked():
                    self.create_window = self.CreateDatasetWindow()
                    self.create_window.bind_rebuild(self.build_dataset_list)
                    self.create_window.show()

                def save_button_clicked():
                    dsf = get_sheet(self.table)
                    save_dataset(dataset_select.currentItem().text(), dsf)

                create_button = QPushButton('New')
                save_button = QPushButton('Save')
                create_button.clicked.connect(create_button_clicked)
                save_button.clicked.connect(save_button_clicked)

                tail = QHBoxLayout()
                tail.addWidget(create_button)
                tail.addWidget(save_button)

                self.addLayout(tail)

            def bind(self, table: QTableWidget):
                self.table = table
                self.table.cellChanged.connect(self.build_dataset_instruments)

            def build_dataset_list(self):
                self.select.clear()
                dataset_list = load_dataset_list()
                for dataset in dataset_list:
                    item = QListWidgetItem(F'{dataset}', self.select)

            def build_dataset_instruments(self, ds_name):
                if ds_name is None:
                    # Zero off self.table
                    pass
                else:
                    add_as_dataset_change(ds_name)
                    df = load_dataset_list(ds_name)
                    # TODO self.table -> df

            class CreateDatasetWindow(QWidget):
                def __init__(self):
                    super().__init__()
                    self.window()
                    self.setWindowTitle('Create new Dataset')
                    self.rebuild_f = None

                def bind_rebuild(self, rebuild_f):
                    self.rebuild_f = rebuild_f

                def window(self):
                    main_layout = QVBoxLayout()
                    body = QHBoxLayout()
                    tail = QHBoxLayout()

                    name = QTextEdit()
                    name_label = QLabel('Name')

                    cancel_button = QPushButton('Cancel')
                    select_button = QPushButton('Select')

                    def create():
                        if name.document().toPlainText():
                            write_new_empty_dataset(F'{name.document().toPlainText()}')
                            back()
                        else:
                            alert = QMessageBox('The name cannot be empty!')
                            alert.show()

                    def back():
                        self.rebuild_f()
                        self.close()

                    cancel_button.clicked.connect(back)
                    select_button.clicked.connect(create)

                    body.addWidget(name_label)
                    body.addWidget(name)

                    tail.addWidget(cancel_button)
                    tail.addWidget(select_button)

                    main_layout.addLayout(body)
                    main_layout.addLayout(tail)

                    self.setLayout(main_layout)

                def back(self):
                    self.close()

        class DatatablePane(QVBoxLayout):
            def __init__(self):
                super().__init__()
                self.table = None
                self.pane()

            def pane(self):
                self.addWidget(QLabel('List of Instruments'))
                self.table = QTableWidget(20, 3)

                # def on_edit(event):
                #     print("On Edit", get_sheet(self.table))
                #
                # self.table.cellChanged.connect(on_edit)
                self.addWidget(self.table)

                def size_up(n: int):
                    pass

    class TradeAdvisorPage(QWidget):

        def __init__(self):
            self.keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
            super().__init__()
            self.window()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Space:
                pass
            print(event.key())

        def window(self):
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

            details_pane = TradeHunterApp.TradeAdvisorPage.TradeAdvisorDetailsPane()
            bottom_body.addLayout(details_pane, 2)

            def on_back_button_pressed():
                back()

            def back():
                window = TradeHunterApp.MainWindow()
                window.show()
                self.close()

            back_button = QPushButton('Back')
            tail.addWidget(back_button)
            back_button.clicked.connect(back)

            layout.addLayout(head)
            layout.addLayout(body)
            layout.addLayout(bottom_body)
            layout.addLayout(tail)

            self.setWindowTitle('Trade Advisors')
            self.setLayout(layout)

        class TradeAdvisorDetailsPane(QVBoxLayout):

            def __init__(self):
                super().__init__()
                self.pane()

            def pane(self):
                self.addWidget(QLabel('Details'))
                details_layout = QHBoxLayout()
                details_layout.addWidget(QLabel('None'))
                self.addLayout(details_layout)

    class TestingChamberPage(QWidget):

        def __init__(self):
            super().__init__()

    class AnalysisWindow(QWidget):

        def __init__(self):
            super().__init__()

    # def create_dataset_window(self):
    #     window = QWidget()
    #     layout = QVBoxLayout()
    #     pass
    #
    # def add_instrument_window(self):
    #     window = QWidget()
    #     layout = QVBoxLayout()
    #     pass
    #
    # # Minor windows
    #
    # def generic_context_window(self, args, previous_window):
    #     window = QWidget()
    #     layout = QVBoxLayout()
    #     self.options = -1
    #
    #     for i in range(args):
    #         def option_click(self):
    #             self.options = i
    #             previous_window.show()
    #             window.destroy()
    #
    #         button = QPushButton(args[i])
    #         button.clicked.connect(option_click)
    #         layout.addWidget(button)
    #
    #     window.setLayout(layout)
    #     window.setWindowTitle('Select an Option')
    #     window.show()
    #
    # def update_all_with_progress_bar(self, title):
    #     """Redownload data according to updated datasets.
    #     Returns to data management window."""
    #     dsc = get_dataset_changes()
    #
    #     window = QWidget()
    #     layout = QBoxLayout()
    #
    #     progress_bar = QProgressBar()
    #     layout.addWidget(progress_bar)
    #
    #     window.setLayout(layout)
    #     window.setWindowTitle(title)
    #     window.show()
    #
    #     l = len(dsc['index'])
    #     progress_bar.setMinimum(0)
    #     progress_bar.setMaximum(l)
    #     for i in range(dsc['index']):
    #         update_specific_dataset_change(dsc['index'][i])
    #         progress_bar.setValue(i)

################ Purpose ################
#
# Create and Modify data set definitions - each containing lists of instruments, intervals and periods (tuple) to
# measure.
# Then, data can be downloaded according to the data set definitions. Finally, robots can be evaluated against
# these set definitions (if they are downloaded) and the results will be saved.
# These results can be viewed and graphics can be loaded for more information.


