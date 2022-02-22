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
from UI.QTUtil import get_datatable_sheet, set_datatable_sheet, clear_table
from util.dataRetrievalUtil import load_trade_advisor_list, get_dataset_changes, update_specific_dataset_change, \
    write_new_empty_dataset, load_dataset_list, save_dataset, add_as_dataset_change, load_dataset, \
    load_symbol_suggestions, load_interval_suggestions, load_period_suggestions
from util.langUtil import normify_name


class TradeHunterApp:

    def __init__(self):
        self.app = QApplication(sys.argv)
        main = self.MainWindow()
        main.show()
        self.app.exec()

    class MainWindow(QWidget):

        def __init__(self):
            self.keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
            super().__init__()

            self.dm_window = None
            self.ta_window = None
            self.window()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_D:
                self.dm_window = TradeHunterApp.DataManagementPage()
                self.dm_window.show()
                self.close()
            elif event.key() == Qt.Key_T:
                self.ta_window = TradeHunterApp.TradeAdvisorPage()
                self.ta_window.show()
                self.close()
            elif event.key() == Qt.Key_Escape:
                self.close()
            print("Main Window keypress", event.key())

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
            print("Data Management Window keypress", event.key())

        def window(self):

            head_body_tail = QVBoxLayout()
            head = QHBoxLayout()
            body = QHBoxLayout()
            tail = QHBoxLayout()
            head_body_tail.addLayout(head)
            head_body_tail.addLayout(body)
            head_body_tail.addLayout(tail)

            def back():
                window = TradeHunterApp.MainWindow()
                window.show()
                self.close()

            left = TradeHunterApp.DataManagementPage.DatasetPane()
            self.datasetpane = left
            right = TradeHunterApp.DataManagementPage.DatatablePane()
            self.datatablepane = right
            left.bind(right.table)

            body.addLayout(left)
            body.addLayout(right)
            body.setStretchFactor(left, 1)
            body.setStretchFactor(right, 1.5)

            update_all_button = QPushButton('Update All')
            back_button = QPushButton('Back')
            back_button.clicked.connect(back)
            tail.addWidget(back_button)
            tail.addWidget(update_all_button)

            self.setLayout(head_body_tail)

        class DatasetPane(QVBoxLayout):

            def __init__(self):
                super().__init__()
                self.saved = True
                self.table = None
                self.select = None
                self.window()

            def window(self):

                dataset_select = QListWidget()
                dataset_label = QLabel('Dataset')

                self.select = dataset_select
                self.build_dataset_list()

                def on_symbol_select(event):
                    print("Symbol", event.text())

                def on_interval_select(event):
                    print("Interval", event.text())

                def on_period_select(event):
                    print("Period", event.text())

                def load_selected_dataset(event):
                    if event:
                        print("Selected ", event.text())
                        self.build_dataset_instruments(event.text())
                    else:
                        print("Selected None")
                        self.build_dataset_instruments(None)
                    self.saved = True

                dataset_select.currentItemChanged.connect(load_selected_dataset)

                select_layout = QHBoxLayout()
                select_layout.addWidget(dataset_label)
                select_layout.addWidget(dataset_select)
                self.addLayout(select_layout)

                mid = QVBoxLayout()

                floor1 = QHBoxLayout()
                floor2 = QHBoxLayout()
                floor3 = QHBoxLayout()

                # Mid left - Labels
                floor1.addWidget(QLabel('Symbols'), 0.5)
                floor2.addWidget(QLabel('Interval'), 0.5)
                floor3.addWidget(QLabel('Period'), 0.5)

                # Mid mid - list widgets
                symbol_list = QListWidget()
                interval_list = QListWidget()
                period_list = QListWidget()

                # Build listwidget items
                for string in load_symbol_suggestions():
                    item = QListWidgetItem(string, symbol_list)
                for string in load_interval_suggestions():
                    item = QListWidgetItem(string, interval_list)
                for string in load_period_suggestions():
                    item = QListWidgetItem(string, period_list)

                symbol_list.currentItemChanged.connect(on_symbol_select)
                interval_list.currentItemChanged.connect(on_interval_select)
                period_list.currentItemChanged.connect(on_period_select)

                floor1.addWidget(symbol_list, 2)
                floor2.addWidget(interval_list, 2)
                floor3.addWidget(period_list, 2)

                mid.addLayout(floor1)
                mid.addLayout(floor2)
                mid.addLayout(floor3)
                self.addLayout(mid)

                def create_button_clicked():
                    self.create_window = self.CreateDatasetWindow()
                    self.create_window.bind_rebuild(self.build_dataset_list)
                    self.create_window.show()

                def save_button_clicked():
                    dsf = get_datatable_sheet(self.table)
                    ds_name = dataset_select.currentItem().text()
                    if self.saved:
                        print("Already up to date, skip saving.")
                        return
                    # Save
                    save_dataset(ds_name, dsf)
                    # Mark as change - updates will re-download all files in dataset marked by "change"
                    add_as_dataset_change(ds_name)
                    # Rebuild instrument list (e.g. Bad rows will be deleted)
                    self.build_dataset_instruments(ds_name)

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
                self.table.cellChanged.connect(self.on_cell_change)
                self.table.currentItemChanged.connect(self.on_cell_change)

            def build_dataset_list(self):
                self.select.clear()
                dataset_list = load_dataset_list()
                for dataset in dataset_list:
                    item = QListWidgetItem(F'{dataset}', self.select)

            def on_cell_change(self, event):
                self.saved = False

            def build_dataset_instruments(self, ds_name):
                if ds_name is None:
                    clear_table(self.table)
                else:
                    print("Building dataset instruments", ds_name)
                    df = load_dataset(ds_name)
                    set_datatable_sheet(self.table, df)

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
                self.table = QTableWidget(100, 3)
                self.addWidget(self.table)

    class TradeAdvisorPage(QWidget):

        def __init__(self):
            self.keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
            super().__init__()
            self.window()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Space:
                pass
            print("Trade Advisor keypress", event.key())

        def window(self):
            layout = QVBoxLayout()

            test_robot_button = QPushButton('Test')
            optimise_robot_button = QPushButton('Optimise')
            optimise_dataset_button = QPushButton('Analyse Data')
            back_button = QPushButton('Back')

            layout.addWidget(test_robot_button)
            layout.addWidget(optimise_robot_button)
            layout.addWidget(optimise_dataset_button)
            layout.addWidget(back_button)

            def on_test_robot_button_pressed():

                def test_robot_window():
                    tr_window = QWidget()

                    select_layout = QHBoxLayout()
                    robot_label = QLabel('Robot')
                    robot_select = QListWidget()
                    select_layout.addWidget(robot_label)
                    select_layout.addWidget(robot_select)

                    for ta in load_trade_advisor_list():
                        item = QListWidgetItem(ta, robot_select)

                    layout = QVBoxLayout()

                    button_layout = QHBoxLayout()
                    confirm_button = QPushButton('Confirm')
                    cancel_button = QPushButton('Cancel')

                    def on_confirm():
                        self.close()
                        if not robot_select.currentItem():
                            QMessageBox('You have not selected a robot')
                        else:
                            print("Select:", robot_select.currentItem().text())
                        self.test_chamber_window = TradeHunterApp.TestingChamberPage()

                    def on_cancel():
                        tr_window.close()

                    confirm_button.clicked.connect(on_confirm)
                    cancel_button.clicked.connect(on_cancel)
                    button_layout.addWidget(cancel_button)
                    button_layout.addWidget(confirm_button)

                    layout.addLayout(select_layout)
                    layout.addLayout(button_layout)

                    tr_window.setLayout(layout)
                    return tr_window

                self.tr_window = test_robot_window()
                self.tr_window.show()

            test_robot_button.clicked.connect(on_test_robot_button_pressed)

            def on_optimise_robot_button_pressed():

                def optimise_robot_window():
                    or_window = QWidget()

                    select_layout = QHBoxLayout()
                    robot_label = QLabel('Robot')
                    robot_select = QListWidget()
                    select_layout.addWidget(robot_label)
                    select_layout.addWidget(robot_select)

                    for ta in load_trade_advisor_list():
                        item = QListWidgetItem(ta, robot_select)

                    layout = QVBoxLayout()

                    button_layout = QHBoxLayout()
                    confirm_button = QPushButton('Confirm')
                    cancel_button = QPushButton('Cancel')

                    def on_confirm():
                        self.close()
                        if not robot_select.currentItem():
                            QMessageBox('You have not selected a robot')
                        else:
                            print("Select:", robot_select.currentItem().text())
                        self.test_chamber_window = TradeHunterApp.TestingChamberPage()

                    def on_cancel():
                        or_window.close()

                    confirm_button.clicked.connect(on_confirm)
                    cancel_button.clicked.connect(on_cancel)
                    button_layout.addWidget(cancel_button)
                    button_layout.addWidget(confirm_button)

                    layout.addLayout(select_layout)
                    layout.addLayout(button_layout)

                    or_window.setLayout(layout)
                    return or_window

                self.or_window = optimise_robot_window()
                self.or_window.show()

            optimise_robot_button.clicked.connect(on_optimise_robot_button_pressed)

            def on_optimise_dataset_button_pressed():

                def optimise_dataset_window():
                    od_window = QWidget()

                    select_layout = QHBoxLayout()
                    robot_label = QLabel('Robot')
                    robot_select = QListWidget()
                    select_layout.addWidget(robot_label)
                    select_layout.addWidget(robot_select)

                    layout = QVBoxLayout()

                    button_layout = QHBoxLayout()
                    confirm_button = QPushButton('Confirm')
                    cancel_button = QPushButton('Cancel')

                    def on_confirm():
                        self.close()
                        if not robot_select.currentItem():
                            QMessageBox('You have not selected a robot')
                        else:
                            print("Select:", robot_select.currentItem().text())
                        self.test_chamber_window = TradeHunterApp.TestingChamberPage()

                    def on_cancel():
                        od_window.close()

                    confirm_button.clicked.connect(on_confirm)
                    cancel_button.clicked.connect(on_cancel)
                    button_layout.addWidget(cancel_button)
                    button_layout.addWidget(confirm_button)

                    layout.addLayout(select_layout)
                    layout.addLayout(button_layout)

                    od_window.setLayout(layout)
                    return od_window

                self.tr_window = optimise_dataset_window()
                self.tr_window.show()

            optimise_dataset_button.clicked.connect(on_optimise_dataset_button_pressed)

            def on_back_button_pressed():
                back()
            back_button.clicked.connect(on_back_button_pressed)

            def back():
                window = TradeHunterApp.MainWindow()
                window.show()
                self.close()

            self.setLayout(layout)

    class TestingChamberPage(QWidget):

        def __init__(self):
            super().__init__()

    class OptimisationPage(QWidget):

        def __init__(self):
            self.window()

        def window(self):
            pass

    class AnalysisWindow(QWidget):

        def __init__(self):
            super().__init__()
