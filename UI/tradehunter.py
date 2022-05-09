import PyQt5
import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
                             QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
                             QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget, QMessageBox, QBoxLayout, QListWidget, QMainWindow, QListWidgetItem,
                             QAbstractItemView)

# Settings
from matplotlib import gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
import sys

from matplotlib.figure import Figure

from UI.QTUtil import get_datatable_sheet, set_datatable_sheet, clear_table, set_col_cell_sheet, get_dataset_table, \
    set_dataset_table
from util.dataGraphingUtil import plot_single, candlestick_plot, init_plot, DATE_FORMAT_DICT, get_interval
from util.dataRetrievalUtil import load_trade_advisor_list, get_dataset_changes, update_specific_dataset_change, \
    write_new_empty_dataset, load_dataset_list, save_dataset, add_as_dataset_change, load_dataset, \
    load_symbol_suggestions, load_interval_suggestions, load_period_suggestions, update_all_dataset_changes, \
    retrieve_ds, clear_dataset_changes, load_df_list, load_df, load_ivar_list, get_test_steps, \
    load_ivar, load_lag_suggestions, load_leverage_suggestions, load_instrument_type_suggestions, \
    load_ivar_as_list, translate_xvar_dict, load_flat_commission_suggestions, load_speed_suggestions, load_ivar_as_dict, \
    load_capital_suggestions, remove_all_df, remove_dataset_change, remove_dataset, delete_ivar, get_random_df, \
    load_optim_depth_suggestions, load_setting, remove_ds_df, load_algo_list, load_algo_ivar_list
from util.dataTestingUtil import step_test_robot, DataTester, write_test_result, write_test_meta, load_test_result, \
    load_test_meta, get_tested_robot_list, get_tests_list, get_ivar_vars
from util.langUtil import normify_name, try_int, leverage_to_float, get_test_name, try_float, strtodatetime, \
    timedeltatoyahootimestr


class TradeHunterApp:

    def __init__(self):
        self.app = QApplication(sys.argv)
        main = self.MainWindow()
        main.show()
        self.app.exec()
        self.fullscreen = True
        self.curr_window = None

    class MainWindow(QWidget):

        def __init__(self):
            self.keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
            super().__init__()

            self.dm_window = None
            self.ta_window = None
            self.sp_window = None
            self.window()

            # self.showFullScreen()

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
            sp_button = QPushButton('General Plotter')
            set_button = QPushButton('Settings')

            def on_dm_click():
                self.dm_window = TradeHunterApp.DataManagementPage()
                self.dm_window.show()
                self.close()

            def on_ta_click():
                self.ta_window = TradeHunterApp.TradeAdvisorPage()
                self.ta_window.show()
                self.close()

            def on_sp_click():
                self.sp_window = TradeHunterApp.GeneralPlotterPage()
                self.sp_window.show()
                self.close()

            dm_button.clicked.connect(on_dm_click)
            ta_button.clicked.connect(on_ta_click)
            sp_button.clicked.connect(on_sp_click)
            layout.addWidget(dm_button)
            layout.addWidget(ta_button)
            layout.addWidget(sp_button)
            layout.addWidget(set_button)

            self.setLayout(layout)
            self.setWindowTitle('Trade Hunter')
            self.show()

    # Main Pages
    class GeneralPlotterPage(QWidget):
        def __init__(self):
            super().__init__()

            self.splot_window = None
            self.ssim_window = None
            self.stest_window = None
            self.rc_window = None
            self.dc_window = None

            self.window()

        def window(self):
            layout = QVBoxLayout()

            splot_button = QPushButton('Simple Plotter')
            ssym_button = QPushButton('1-Symbol Test Plot')
            ssim_button = QPushButton('Single Sim')
            # stest_button = QPushButton('Single Test')
            # rc_button = QPushButton('Robot Comparison')
            # dc_button = QPushButton('Data Comparison')
            back_button = QPushButton('Back')

            body = QVBoxLayout()

            body.addWidget(splot_button)
            body.addWidget(ssim_button)
            body.addWidget(ssym_button)
            # body.addWidget(stest_button)
            # body.addWidget(rc_button)
            # body.addWidget(dc_button)
            body.addWidget(back_button)

            tail = QVBoxLayout()
            button_layout = QHBoxLayout()

            def splot_clicked():
                self.splot_window = TradeHunterApp.SimplePlotter()
                self.splot_window.show()

            def ssim_clicked():
                self.ssim_window = TradeHunterApp.SimPlotter()
                self.ssim_window.show()

            def ssym_clicked():
                """Single Symbol test: Tests robot against
                a single datafile and plots the results."""
                pass
                # Candlestick -> Line plot

                # Equity Plot

                # Trade deals (No profit-loss boxes)

            def stest_clicked():
                self.ssim_window = TradeHunterApp.SimPlotter()
                self.ssim_window.show()

            def rc_button_clicked():
                self.ssim_window = TradeHunterApp.SimPlotter()
                self.ssim_window.show()

            def dc_button_clicked():
                self.ssim_window = TradeHunterApp.SimPlotter()
                self.ssim_window.show()

            # This window does not close upon plotting.
            splot_button.clicked.connect(splot_clicked)
            ssim_button.clicked.connect(ssim_clicked)
            # stest_button.clicked.connect(stest_clicked)
            # rc_button.clicked.connect(rc_button_clicked)
            # dc_button.clicked.connect(dc_button_clicked)
            back_button.clicked.connect(self.back)

            button_layout.addWidget(back_button)
            tail.addLayout(button_layout)

            layout.addLayout(body)
            layout.addLayout(tail)

            self.setLayout(layout)
            self.show()

        def back(self):
            window = TradeHunterApp.MainWindow()
            window.show()
            self.close()

    class DataManagementPage(QWidget):

        def __init__(self):
            self.keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
            super().__init__()
            self.windowTitle = 'Data Management'

            self.p_window = None
            self.download_progress = None
            self.datasetpane = None
            self.datatablepane = None
            self.main_buttons = []

            self.window()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Space:
                pass
            elif event.key() == Qt.Key_Escape:
                pass
            elif event.key() == Qt.Key_Enter:
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

            def progress_window(n_files: int) -> QProgressBar:
                print("Update Progress Bar initialised!")
                _progress_window = QWidget()

                p_layout = QVBoxLayout()
                download_progress = QProgressBar()
                download_progress.setMinimum(0)
                download_progress.setMaximum(n_files)
                download_progress.setValue(0)
                p_layout.addWidget(download_progress)

                _progress_window.setLayout(p_layout)

                self.p_window = _progress_window
                self.download_progress = download_progress
                self.p_window.show()

                return _progress_window, download_progress

            def update_all():
                print("----------------------------------")
                print("Preparing to update changed files")

                files_df = get_dataset_changes()

                self.p_window, self.d_progress = progress_window(len(files_df.index))
                self.p_window.showMinimized()
                self.p_window.show()

                # Temporarily disable buttons
                for button in self.main_buttons:
                    button.setEnabled(False)

                print("Updating...")
                for index, row in files_df.iterrows():
                    self.download_progress.setValue(self.download_progress.value() + 1)
                    self.p_window.setWindowTitle(F"Downloading {row['name']}")
                    success = retrieve_ds(row['name'], True)
                    PyQt5.QtWidgets.QApplication.processEvents()

                    # If failed, delete dataset change
                    remove_dataset_change(row['name'])

                # Reenable buttons
                for button in self.main_buttons:
                    button.setEnabled(True)

                self.p_window.setWindowTitle(F"Download complete. You may close the window.")

                clear_dataset_changes()

                # In case of change, update dataset(s) display
                left.build_dataset_instruments(left.combo.currentText())

            def import_clicked():
                pass

            def clean_all():
                ds_names = []
                for ds_name in ds_names:
                    # Mark all datasets as un-updated
                    add_as_dataset_change(ds_name)
                remove_all_df()

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

            # Window buttons
            back_button = QPushButton('Back')
            import_button = QPushButton('Import')
            update_all_button = QPushButton('Update All')
            clean_all_button = QPushButton('Clean All')

            # Add button handles class-wide
            self.main_buttons.append(back_button)
            self.main_buttons.append(import_button)
            self.main_buttons.append(update_all_button)
            self.main_buttons.append(clean_all_button)

            # Connect events
            back_button.clicked.connect(back)
            import_button.clicked.connect(import_clicked)
            update_all_button.clicked.connect(update_all)
            clean_all_button.clicked.connect(clean_all)

            progress_bar = QProgressBar()
            progress_bar.setWindowTitle("Progress Bar")

            tail.addWidget(back_button)
            tail.addWidget(import_button)
            tail.addWidget(update_all_button)
            tail.addWidget(clean_all_button)

            self.setLayout(head_body_tail)

        class DatasetPane(QVBoxLayout):

            def __init__(self):
                super().__init__()
                self.saved = True
                self.table = None
                self.select = None
                self.combo = None
                self.c_win = None

                self.window()

            def window(self):

                # dataset_select = QListWidget()
                dataset_combo = QComboBox()
                # dataset_select.setFixedHeight(20)
                dataset_label = QLabel('Dataset')

                self.combo = dataset_combo
                # self.select = dataset_select
                self.build_dataset_list()

                def on_symbol_select(event):
                    # print("Symbol", event.text())
                    print("Symbol", symbol_combo.currentText())
                    add_table_text(symbol_combo.currentText(), 0)

                def on_interval_select(event):
                    # print("Interval", event.text())
                    print("Interval", interval_combo.currentText())
                    add_table_text(interval_combo.currentText(), 1)

                def on_period_select(event):
                    # print("Period", event.text())
                    print("Period", period_combo.currentText())
                    add_table_text(period_combo.currentText(), 2)

                def add_table_text(string, colIdx):
                    set_col_cell_sheet(self.table, string, colIdx)

                def load_combo_dataset(event):
                    if event > -1:
                        self.build_dataset_instruments(self.combo.currentText())

                # dataset_select.currentItemChanged.connect(load_selected_dataset)
                dataset_combo.currentIndexChanged.connect(load_combo_dataset)

                select_layout = QHBoxLayout()
                select_layout.addWidget(dataset_label)
                select_layout.addWidget(dataset_combo)
                # select_layout.addWidget(dataset_select)
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

                symbol_combo = QComboBox()
                interval_combo = QComboBox()
                period_combo = QComboBox()

                # Build listwidget items
                si = 0
                for string in load_symbol_suggestions():
                    # item = QListWidgetItem(string, symbol_list)
                    symbol_combo.insertItem(si, string)
                    si += 1
                si = 0
                for string in load_interval_suggestions():
                    # item = QListWidgetItem(string, interval_list)
                    interval_combo.insertItem(si, string)
                    si += 1
                si = 0
                for string in load_period_suggestions():
                    # item = QListWidgetItem(string, period_list)
                    period_combo.insertItem(si, string)
                    si += 1
                symbol_combo.setCurrentIndex(0)
                interval_combo.setCurrentIndex(0)
                period_combo.setCurrentIndex(0)

                # symbol_list.setFixedHeight(20)
                # interval_list.setFixedHeight(20)
                # period_list.setFixedHeight(20)

                symbol_list.currentItemChanged.connect(on_symbol_select)
                interval_list.currentItemChanged.connect(on_interval_select)
                period_list.currentItemChanged.connect(on_period_select)

                symbol_combo.currentIndexChanged.connect(on_symbol_select)
                interval_combo.currentIndexChanged.connect(on_interval_select)
                period_combo.currentIndexChanged.connect(on_period_select)

                # floor1.addWidget(symbol_list, 2)
                # floor2.addWidget(interval_list, 2)
                # floor3.addWidget(period_list, 2)
                floor1.addWidget(symbol_combo, 2)
                floor2.addWidget(interval_combo, 2)
                floor3.addWidget(period_combo, 2)

                mid.addLayout(floor1)
                mid.addLayout(floor2)
                mid.addLayout(floor3)
                self.addLayout(mid)

                def create_button_clicked():
                    self.create_window = self.CreateDatasetWindow()
                    self.create_window.bind_rebuild(self.build_dateset_specific)
                    self.create_window.show()

                def save_button_clicked():
                    dsf = get_datatable_sheet(self.table)
                    ds_name = dataset_combo.currentText()
                    if self.saved:
                        print("Already up to date, skip saving.")
                        return
                    # Save
                    save_dataset(ds_name, dsf)
                    self.saved = True
                    # Mark as change - updates will re-download all files in dataset marked by "change"
                    add_as_dataset_change(ds_name)
                    # Rebuild instrument list (e.g. Bad rows will be deleted)
                    self.build_dataset_instruments(ds_name)

                def delete_button_clicked():
                    ds_name = dataset_combo.currentText()
                    layout_list = QVBoxLayout()

                    # Construct confirm window
                    confirm_window = QWidget()
                    confirm_button = QPushButton('Delete')
                    cancel_button = QPushButton('Cancel')
                    text_label = QLabel(F'Delete {ds_name}?')
                    text_layout = QHBoxLayout()
                    text_layout.addWidget(text_label)
                    confirm_layout = QHBoxLayout()

                    # Refresh
                    self.build_dataset_list()

                    def cancel():
                        self.c_win.close()

                    def confirm():
                        remove_dataset_change(ds_name)
                        remove_ds_df(ds_name)
                        remove_dataset(ds_name)
                        self.c_win.close()

                    confirm_button.clicked.connect(confirm)
                    cancel_button.clicked.connect(cancel)

                    layout_list.addLayout(text_layout)
                    layout_list.addLayout(confirm_layout)
                    confirm_window.setLayout(layout_list)
                    confirm_layout.addWidget(confirm_button)
                    confirm_layout.addWidget(cancel_button)

                    confirm_window.setWindowTitle(F'Dataset deletion')

                    self.c_win = confirm_window
                    confirm_window.show()

                create_button = QPushButton('New')
                save_button = QPushButton('Save')
                delete_button = QPushButton('Delete')
                create_button.clicked.connect(create_button_clicked)
                save_button.clicked.connect(save_button_clicked)
                delete_button.clicked.connect(delete_button_clicked)

                tail = QHBoxLayout()
                tail.addWidget(create_button)
                tail.addWidget(save_button)
                tail.addWidget(delete_button)

                self.addLayout(tail)
                self.saved = True

            def bind(self, table: QTableWidget):
                self.table = table
                self.build_dataset_instruments(self.combo.currentText())
                self.table.cellChanged.connect(self.on_cell_change)
                self.table.currentItemChanged.connect(self.on_cell_change)

            def build_dataset_list(self):
                dataset_list = load_dataset_list()
                i = 0
                self.combo.clear()
                for dataset in dataset_list:
                    # item = QListWidgetItem(F'{dataset}', self.select)
                    self.combo.insertItem(i, F'{dataset}')
                    i += 1
                self.combo.setCurrentIndex(0)

            def build_dateset_specific(self, ds_name: str):
                dataset_list = load_dataset_list()
                i = 0
                self.combo.clear()
                for dataset in dataset_list:
                    # item = QListWidgetItem(F'{dataset}', self.select)
                    self.combo.insertItem(i, F'{dataset}')
                    if dataset == ds_name:
                        self.combo.setCurrentIndex(i)
                    i += 1

            def on_cell_change(self, event):
                self.saved = False

            def build_dataset_instruments(self, ds_name):
                if ds_name is None or ds_name == '':
                    clear_table(self.table)
                else:
                    print("Building dataset instruments", ds_name)
                    df = load_dataset(ds_name)
                    set_datatable_sheet(self.table, df)

            class CreateDatasetWindow(QWidget):

                def __init__(self):
                    self.keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
                    super().__init__()

                    # Handles
                    self.name = None

                    self.window()
                    self.setWindowTitle('Create new Dataset')
                    self.rebuild_f = None

                def keyPressEvent(self, event):
                    if event.key() == Qt.Key_Space:
                        pass
                    elif event.key() == Qt.Key_Enter:
                        self.create()
                    # Couldn't get it to work.
                    print("Create Dataset keypress", event.key())

                def bind_rebuild(self, rebuild_f):
                    self.rebuild_f = rebuild_f

                def create(self):
                    if self.name.document().toPlainText():
                        _name = self.name.document().toPlainText() + '.csv'
                        write_new_empty_dataset(F'{_name}')
                        self.rebuild_f(_name)
                        # self.back()  # do not trigger back()'s rebuild_f('')
                        self.close()
                    else:
                        self.alert_window = QWidget()
                        alert = QMessageBox(self.alert_window)
                        alert.setText('The name cannot be empty!')
                        alert.show()

                def window(self):
                    main_layout = QVBoxLayout()
                    body = QHBoxLayout()
                    tail = QHBoxLayout()

                    self.name = QTextEdit()
                    self.name.setFixedHeight(20)
                    name_label = QLabel('Name')

                    cancel_button = QPushButton('Cancel')
                    select_button = QPushButton('Select')

                    def back():
                        self.rebuild_f('')
                        self.close()

                    cancel_button.clicked.connect(back)
                    select_button.clicked.connect(self.create)

                    body.addWidget(name_label)
                    body.addWidget(self.name)

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
                self.table.setHorizontalHeaderLabels(['Symbol', 'Interval', 'Period'])
                self.addWidget(self.table)

    class TradeAdvisorPage(QWidget):

        def __init__(self):
            self.keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
            super().__init__()

            # pages
            self.tr_window = None
            self.or_window = None
            self.rr_window = None
            self.od_window = None

            self.window()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Space:
                pass
            print("Trade Advisor keypress", event.key())

        def window(self):
            layout = QVBoxLayout()

            test_robot_button = QPushButton('Testing Chamber')
            robot_results_button = QPushButton('Results Analysis')
            back_button = QPushButton('Back')

            layout.addWidget(test_robot_button)
            # layout.addWidget(optimise_robot_button)
            layout.addWidget(robot_results_button)
            # layout.addWidget(optimise_dataset_button)
            layout.addWidget(back_button)

            def on_test_robot_button_pressed():

                def test_robot_window():

                    tr_window = QWidget()

                    select_layout = QHBoxLayout()
                    robot_label = QLabel('Robot')
                    # robot_select = QListWidget()
                    # robot_select.setFixedHeight(20)
                    robot_combo = QComboBox()
                    select_layout.addWidget(robot_label)
                    select_layout.addWidget(robot_combo)

                    ta_list = load_trade_advisor_list()
                    ta_list.sort(reverse=True)
                    for ta in ta_list:
                        # item = QListWidgetItem(ta, robot_select)
                        robot_combo.insertItem(0, ta)
                    robot_combo.setCurrentIndex(0)

                    layout = QVBoxLayout()

                    button_layout = QHBoxLayout()
                    confirm_button = QPushButton('Confirm')
                    cancel_button = QPushButton('Cancel')

                    def on_confirm():
                        if not robot_combo.currentText():
                            self.alert_window = QWidget()
                            alert = QMessageBox(self.alert_window)
                            alert.setText('You have not selected a robot!')
                            alert.show()
                        else:
                            # print("Select:", robot_select.currentItem().text())
                            # self.test_chamber_window = TradeHunterApp.TestingChamberPage(
                            #     robot_select.currentItem().text())
                            print("Select:", robot_combo.currentText())
                            self.test_chamber_window = TradeHunterApp.TestingChamberPage(
                                robot_combo.currentText())
                            self.test_chamber_window.show()
                        on_cancel()

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

            def on_robot_results_button_pressed():

                def results_analysis_window():
                    # link to result analysis page
                    result_analysis_window = TradeHunterApp.ResultAnalysisPage('default', 'default')
                    return result_analysis_window

                self.rr_window = results_analysis_window()
                self.rr_window.show()

            robot_results_button.clicked.connect(on_robot_results_button_pressed)

            def on_back_button_pressed():
                back()

            back_button.clicked.connect(on_back_button_pressed)

            def back():
                window = TradeHunterApp.MainWindow()
                window.show()
                self.close()

            self.setLayout(layout)

    # - Trade Advisor Pages
    class TestingChamberPage(QWidget):

        def __init__(self, robot_name, prev_window=None):
            super().__init__()
            self.robot_name = robot_name
            self.setWindowTitle(robot_name)
            self.window()

            # progress
            self.p_window = None
            self.p_bar = None
            self.p_bar_embed = None

            # graph
            self.canvas = None

            # window handles
            self.ta_window = None
            self.prev_window = prev_window
            self.next_window = None

        def window(self):

            whole = QVBoxLayout()

            head = QHBoxLayout()
            body = QHBoxLayout()
            tail = QHBoxLayout()

            # ===============================================================
            #   whole
            #       head
            #           X
            #       body
            #           panes
            #               left_pane (dataset, table, ivar)
            #               xvar_pane (xvar)
            #       tail
            # ===============================================================

            panes = QHBoxLayout()
            left_pane = QVBoxLayout()

            # === Left Pane ===
            dataset_layout = QHBoxLayout()
            ivar_layout = QHBoxLayout()
            xvar_pane = QHBoxLayout()

            # Build Dataset selector and table
            dataset_label = QLabel('Dataset')
            dataset_combo = QComboBox()
            dataset_list = load_dataset_list()
            dataset_list.sort(reverse=True)
            for ds_name in dataset_list:
                dataset_combo.insertItem(0, ds_name)
            dataset_combo.setCurrentIndex(0)
            dataset_layout.addWidget(dataset_label)
            dataset_layout.addWidget(dataset_combo)

            # Build dataset clear button and table
            dataset_clear_button = QPushButton('Clear')
            dataset_layout.addWidget(dataset_clear_button)

            dataset_table = QTableWidget(100, 1)
            dataset_table.setHorizontalHeaderLabels(['Dataset'])
            dataset_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

            # Dataset combo adds dataset into table on select
            def add_dataset():
                print(F'Adding {dataset_combo.currentText()} into {dataset_table}')
                ds_names = get_dataset_table(dataset_table)
                ds_names.append(dataset_combo.currentText())
                set_dataset_table(dataset_table, ds_names)

            def clear_datasets():
                clear_table(dataset_table)

            # dataset_combo.currentIndexChanged.connect(add_dataset)
            dataset_combo.activated.connect(add_dataset)
            dataset_clear_button.clicked.connect(clear_datasets)

            # Choose ivar
            ivar_label = QLabel('Initial Variables')
            ivar_combo = QComboBox()
            ivar_list = load_ivar_list(self.robot_name)
            ivar_list.sort(reverse=True)
            for ivar in ivar_list:
                ivar_combo.insertItem(0, ivar)
            ivar_layout.addWidget(ivar_label)
            ivar_layout.addWidget(ivar_combo)

            left_pane.addLayout(dataset_layout)
            left_pane.addWidget(dataset_table)
            left_pane.addLayout(ivar_layout)

            left_tail_layout = QHBoxLayout()
            back_button = QPushButton("Back")
            test_button = QPushButton("Test")
            optimise_button = QPushButton("Optimise")
            simulate_button = QPushButton("Simulate")
            delete_button = QPushButton("Delete IVar")

            robot_name = self.robot_name

            def progress_bar_window():
                p_window = QProgressBar()
                return p_window

            def check_input():
                if not ivar_combo.currentText() or not dataset_combo.currentText():
                    self.alert_window = QWidget()
                    alert_layout = QVBoxLayout()
                    alert = QMessageBox(self.alert_window)
                    alert.setText('IVar or Dataset not selected!')
                    alert.show()

                    alert_layout.addWidget(alert)

                    self.alert_window.setLayout(alert_layout)
                    return False

                if not lag_combo.currentText() or not commission_combo.currentText() or \
                        not leverage_combo.currentText() or not instrument_combo.currentText() or \
                        not type_combo.currentText():
                    self.alert_window = QWidget()
                    alert_layout = QVBoxLayout()
                    alert = QMessageBox(self.alert_window)
                    alert.setText('XVar not completed!')
                    alert.show()

                    alert_layout.addWidget(alert)

                    self.alert_window.setLayout(alert_layout)
                    return False

                if type_combo.currentText() == "Multi":
                    print("Multi type selected. Data in dataset must have same period.")

                name_text.setPlainText(normify_name(name_text.document().toPlainText()))

                if not name_text.document().toPlainText() or len(name_text.document().toPlainText()) < 2:
                    self.alert_window = QWidget()
                    alert_layout = QVBoxLayout()
                    alert = QMessageBox(self.alert_window)
                    alert.setText('The test needs a name!')
                    alert.show()

                    alert_layout.addWidget(alert)

                    self.alert_window.setLayout(alert_layout)
                    return False

                # if not commission_combo.document().toPlainText():
                #     capital_text.document().setPlaintText('0')
                if not capital_text.document().toPlainText():
                    capital_text.document().setPlaintText('0')

                capital = try_int(capital_text.document().toPlainText())
                if capital <= 0:
                    self.alert_window = QWidget()
                    alert_layout = QVBoxLayout()
                    alert = QMessageBox(self.alert_window)
                    alert.setText('Please enter a valid number, e.g. 10,000 (USD)')
                    alert.show()

                    alert_layout.addWidget(alert)

                    self.alert_window.setLayout(alert_layout)
                    return False

                return True

            def to_test():

                if not check_input():
                    return

                # Get name
                test_name = name_text.document().toPlainText()

                # Get XVar
                xvar = {'lag': lag_combo.currentText(),
                        'capital': try_int(capital_text.document().toPlainText()),
                        'leverage': leverage_combo.currentText(),
                        'instrument_type': instrument_combo.currentText(),
                        # test specific
                        'test_type': type_combo.currentText()}
                xvar = translate_xvar_dict(xvar)

                ivar_name = ivar_combo.currentText()

                p_window = QWidget()
                p_layout = QVBoxLayout()

                p_bar = progress_bar_window()
                p_window.setLayout(p_layout)
                p_layout.addWidget(p_bar)

                p_window.show()

                ivar = load_ivar_as_dict(robot_name, ivar_name)

                # Setup robot
                data_tester = DataTester(xvar)
                data_tester.bind_progress_bar(p_bar)

                ds_names = get_dataset_table(dataset_table)
                test_result, test_meta = data_tester.test(robot_name, ivar, ds_names, test_name)
                # Test Result and Meta are saved inside .test()

                # Move to Results
                self.rap = TradeHunterApp.ResultAnalysisPage(robot_name, test_name)
                # self.rap.force_load(test_name, robot_name)
                self.rap.show()
                self.close()

            def to_optimise():
                if not check_input():
                    return

                optim_name = name_text.document().toPlainText()

                # Get XVar
                xvar = {'lag': lag_combo.currentText(),
                        'capital': try_int(capital_text.document().toPlainText()),
                        'leverage': leverage_combo.currentText(),
                        'instrument_type': instrument_combo.currentText(),
                        # test specific
                        'test_type': type_combo.currentText()}
                xvar = translate_xvar_dict(xvar)
                # Try to insert optimisation xvar
                if optimisation_combo.currentText():
                    xvar.update({
                        'optim_depth': try_int(optimisation_combo.currentText()),
                    })

                ivar_name = ivar_combo.currentText()

                # p_window = QWidget()
                # p_layout = QVBoxLayout()
                #
                p_bar = progress_bar_window()
                self.canvas = TradeHunterApp.MplMultiCanvas(self, 5, 4, 100, 1, 1)
                PyQt5.QtWidgets.QApplication.processEvents()
                tail.addWidget(p_bar)
                tail.addWidget(self.canvas)

                ivar = load_ivar_as_dict(robot_name, ivar_name)

                # Setup robot
                data_tester = DataTester(xvar)
                data_tester.bind_progress_bar_2(p_bar)

                ds_names = get_dataset_table(dataset_table)
                data_tester.optimise(robot_name, ivar, ds_names, optim_name, True, self.canvas)

                p_bar.deleteLater()

                self.oap = TradeHunterApp.OptimisationAnalysisPage(robot_name, optim_name)
                self.oap.show()
                self.close()

            def to_simulate():

                if not check_input():
                    return

                # Get XVar
                xvar = {'lag': lag_combo.currentText(),
                        'capital': try_int(capital_text.document().toPlainText()),
                        'leverage': leverage_combo.currentText(),
                        'instrument_type': instrument_combo.currentText(),
                        # test specific
                        'test_type': type_combo.currentText()}
                xvar = translate_xvar_dict(xvar)

                ivar_name = ivar_combo.currentText()
                ivar = load_ivar_as_dict(robot_name, ivar_name)

                # Use random df from first dataset selected
                ds_names = get_dataset_table(dataset_table)
                df_name = get_random_df(ds_names[0])
                print(F'Simulating first dataframe {df_name}')

                sim_plot_window = TradeHunterApp.SimPlotter()
                sim_plot_window.test_outside(xvar, ivar, robot_name, df_name, "robot")
                sim_plot_window.show()
                self.next_window = sim_plot_window
                print(F'Moving to sim_plot window')

                self.close()

            def to_delete_ivar():

                ivar_name = ivar_combo.currentText()

                # Cannot delete default
                if "*default" in ivar_name.lower():
                    self.alert_window = QWidget()
                    alert = QMessageBox(self.alert_window)
                    alert.setText('Default IVar cannot be deleted')
                    alert.show()
                    return

                confirm_window = QWidget()
                confirm_layout = QHBoxLayout()

                confirm_button = QPushButton('Yes')
                cancel_button = QPushButton('Cancel')

                confirm_layout.addWidget(confirm_button)
                confirm_layout.addWidget(cancel_button)
                confirm_window.setLayout(confirm_layout)

                def confirm():
                    delete_ivar(robot_name, ivar_name)

                    # reset combo box
                    ivar_combo.clear()
                    for ivar in load_ivar_list(self.robot_name):
                        ivar_combo.insertItem(0, ivar)
                    confirm_window.close()

                def cancel():
                    confirm_window.close()

                confirm_button.clicked.connect(confirm)
                cancel_button.clicked.connect(cancel)

                confirm_window.show()

            # === Bottom === Tail buttons
            test_button.clicked.connect(to_test)
            optimise_button.clicked.connect(to_optimise)
            simulate_button.clicked.connect(to_simulate)
            delete_button.clicked.connect(to_delete_ivar)
            back_button.clicked.connect(self.back)

            left_tail_layout.addWidget(test_button)
            left_tail_layout.addWidget(optimise_button)
            left_tail_layout.addWidget(simulate_button)
            left_tail_layout.addWidget(delete_button)
            left_tail_layout.addWidget(back_button)

            left_pane.addLayout(left_tail_layout)

            # === Right === (xvar attributes)
            name_label = QLabel('Test Name')
            name_text = QTextEdit()
            name_text.setFixedHeight(20)
            lag_label = QLabel('Lag')
            lag_combo = QComboBox()
            capital_label = QLabel('Capital')
            capital_text = QTextEdit()
            capital_text.setFixedHeight(20)
            leverage_label = QLabel('Leverage')
            leverage_combo = QComboBox()
            instrument_label = QLabel('Instrument')
            instrument_combo = QComboBox()
            type_label = QLabel('Type')  # Singular/Multi
            type_combo = QComboBox()
            commission_label = QLabel('Commission')
            commission_combo = QComboBox()
            commission_combo.setFixedHeight(20)
            # Optimisation only
            optimisation_depth_label = QLabel('Optim. Depth')
            optimisation_combo = QComboBox()
            optimisation_combo.setFixedHeight(20)

            test_types = ['Multi', 'Single']
            lag_types = load_lag_suggestions()
            leverage_types = load_leverage_suggestions()
            instrument_types = load_instrument_type_suggestions()
            commission_types = load_flat_commission_suggestions()
            optim_depth_types = load_optim_depth_suggestions()

            # Fill in xvar combo options
            for type in test_types:
                type_combo.insertItem(0, type)
            for type in lag_types:
                lag_combo.insertItem(0, type)
            for type in leverage_types:
                leverage_combo.insertItem(0, type)
            for type in instrument_types:
                instrument_combo.insertItem(0, type)
            for type in commission_types:
                commission_combo.insertItem(0, str(type))
            for type in optim_depth_types:
                optimisation_combo.insertItem(0, str(type))
            type_combo.setCurrentIndex(0)
            lag_combo.setCurrentIndex(0)
            leverage_combo.setCurrentIndex(0)
            instrument_combo.setCurrentIndex(0)
            commission_combo.setCurrentIndex(0)
            optimisation_combo.setCurrentIndex(0)

            # Labels on the left, Widgets on the right
            xvar_left_body = QVBoxLayout()
            xvar_right_body = QVBoxLayout()

            xvar_left_body.addWidget(name_label, 1)
            xvar_left_body.addWidget(lag_label, 1)
            xvar_left_body.addWidget(capital_label, 1)
            xvar_left_body.addWidget(leverage_label, 1)
            xvar_left_body.addWidget(instrument_label, 1)
            xvar_left_body.addWidget(type_label, 1)
            xvar_left_body.addWidget(commission_label, 1)
            xvar_left_body.addWidget(optimisation_depth_label, 1)

            xvar_right_body.addWidget(name_text, 1.5)
            xvar_right_body.addWidget(lag_combo, 1.5)
            xvar_right_body.addWidget(capital_text, 1.5)
            xvar_right_body.addWidget(leverage_combo, 1.5)
            xvar_right_body.addWidget(instrument_combo, 1.5)
            xvar_right_body.addWidget(type_combo, 1.5)
            xvar_right_body.addWidget(commission_combo, 1.5)
            xvar_right_body.addWidget(optimisation_combo, 1.5)

            xvar_pane.addLayout(xvar_left_body)
            xvar_pane.addLayout(xvar_right_body)

            # Bottom progress_bar
            self.p_bar_embed = QProgressBar()

            # Add panes
            panes.addLayout(left_pane)
            panes.addLayout(xvar_pane)
            body.addLayout(panes)

            whole.addLayout(head)
            whole.addLayout(body)
            whole.addLayout(tail)

            self.setLayout(whole)
            self.show()

        def back(self):
            # self.ta_window = TradeHunterApp.TradeAdvisorPage()
            # self.ta_window.show()
            self.close()

        class RobotSetupWindow(QWidget):

            def __init__(self, robot):
                super().__init__()
                self.window()
                self.robot = robot

            def window(self):
                ivar_select = QListWidget()
                dataset_select = QListWidget()

                ivar_select.setFixedHeight(20)
                dataset_select.setFixedHeight(20)

        class CreateIVarWindow(QWidget):
            """Create custom IVars (Parent: TestingChamberPage)"""
            # todo
            def __init__(self, ta_name):
                super().__init__()
                self.robot = ta_name
                self.ivar_combos = []  # discrete/enum {'name': value, 'combo': _combo}
                self.ivar_texts = []  # continuous

                self.window()

            def window(self):

                main_layout = QVBoxLayout()
                ivar_layout = QHBoxLayout()
                button_layout = QHBoxLayout()

                def back():
                    self.close()

                create_button = QPushButton('Create')
                cancel_button = QPushButton('Cancel')
                create_button.clicked.connect(self.create_ivar)
                cancel_button.clicked.connect(back)

                button_layout.addWidget(create_button)
                button_layout.addWidget(cancel_button)

                main_layout.addLayout(ivar_layout)
                main_layout.addLayout(button_layout)

                self.setWindowTitle(F'{self.robot}: Create new IVar')
                self.setLayout(main_layout)

                self.show()

            def load_ivar_options(self):
                args_dict = get_ivar_vars(self.robot)

            def create_ivar_option(self):
                pass

            def create_ivar(self):
                for ivar_combo in self.ivar_combos:
                    pass

                for ivar_text in self.ivar_texts:
                    pass

    class RobotAnalysis(QWidget):

        def __init__(self):
            self.window()
            self.show()

        def window(self):
            vis_button = QPushButton('Visualise')  # Load graphs

        # delete ivars etc

    class DataAnalysis(QWidget):
        """Lets you study details of (single) data and lets you carve them based on their properties e.g.
        ranging versus trending."""
        pass

    class ResultAnalysisPage(QWidget):

        def __init__(self, robot_name='default', test_name='default'):
            super().__init__()

            self.alert_window = None
            self.test_name = ""
            self.test_result = {}
            self.test_meta = {}
            self.summary_dict = {}

            # Interact-able UI Elements:
            self.robot_combo = None
            self.test_combo = None
            self.optim_combo = None
            self.labels = []  # for deletion and re-addition
            self.labels_2 = []
            self.layouts = []
            self.d_p = None
            self.w_p = None

            self.canvas = None
            self.window()
            if robot_name.lower() == 'default' or test_name.lower() == 'default':
                # Default Test Result, not Robot Ivar
                self.summary_dict = {
                    'period': 0,
                    'n_bars': 0,
                    'ticks': 0,
                    #
                    'total_profit': 0,
                    'gross_profit': 0,
                    'gross_loss': 0,
                    #
                    'profit_factor': 0,
                    'recovery_factor': 0,
                    'AHPR': 0,
                    'GHPR': 0,
                    #
                    'total_trades': 0,
                    'total_deals': 0,
                    #
                    'balance_drawdown_abs': 0,
                    'balance_drawdown_max': 0,
                    'balance_drawdown_rel': 0,
                    'balance_drawdown_avg': 0,
                    'balance_drawdown_len_avg': 0,
                    'balance_drawdown_len_max': 0,
                    'equity_drawdown_abs': 0,
                    'equity_drawdown_max': 0,
                    'equity_drawdown_rel': 0,
                    'equity_drawdown_avg': 0,
                    'equity_drawdown_len_avg': 0,
                    'equity_drawdown_len_max': 0,
                    #
                    'expected_payoff': 0,
                    'sharpe_ratio': 0,
                    'standard_deviation': 0,
                    'LR_correlation': 0,
                    'LR_standard_error': 0,
                    #
                    'total_short_trades': 0,
                    'total_long_trades': 0,
                    'short_trades_won': 0,
                    'long_trades_won': 0,
                    'trades_won': 0,
                    'trades_lost': 0,
                    #
                    'largest_profit_trade': 0,
                    'average_profit_trade': 0,
                    'largest_loss_trade': 0,
                    'average_loss_trade': 0,
                    #
                    'longest_trade_length': 0,
                    'shortest_trade_length': 0,
                    'average_trade_length': 0,
                    'average_profit_length': 0,
                    'average_loss_length': 0,
                    'period_to_profit': 0,
                    'period_to_gross': 0,
                    #
                    'max_consecutive_wins': 0,
                    'max_consecutive_profit': 0,
                    'avg_consecutive_wins': 0,
                    'avg_consecutive_profit': 0,
                    'max_consecutive_losses': 0,
                    'max_consecutive_loss': 0,
                    'avg_consecutive_losses': 0,
                    'avg_consecutive_loss': 0,
                    #
                    'n_symbols': 0,
                    'margin_level': 0,
                    'z_score': 0,
                    #
                    'dataset': 'None',
                }
                self.create_labels()
            else:
                self.force_load(robot_name, test_name)

        def window(self):

            # Panes laid out vertically
            main_layout = QVBoxLayout()

            head_layout = QVBoxLayout()
            body_layout = QVBoxLayout()
            tail_layout = QVBoxLayout()

            main_layout.addLayout(head_layout)
            main_layout.addLayout(body_layout)
            main_layout.addLayout(tail_layout)

            # Choice Pane
            choice_pane = QHBoxLayout()
            left_choice = QVBoxLayout()
            right_choice = QVBoxLayout()
            self.robot_combo = QComboBox()
            self.test_combo = QComboBox()
            self.optim_combo = QComboBox()
            self.robot_combo.setFixedHeight(20)
            self.test_combo.setFixedHeight(20)
            self.optim_combo.setFixedHeight(20)
            robot_label = QLabel('Select Robot')
            test_label = QLabel('Select Test')
            left_choice.addWidget(robot_label)
            left_choice.addWidget(test_label)
            right_choice.addWidget(self.robot_combo)
            right_choice.addWidget(self.test_combo)
            choice_pane.addLayout(left_choice)
            choice_pane.addLayout(right_choice)
            head_layout.addLayout(choice_pane)

            quit_button = QPushButton('Exit')
            quit_button.clicked.connect(self.back)
            tail_layout.addWidget(quit_button)

            self.robot_combo_update()
            # load_tests([])  # Wait for selection

            self.robot_combo.activated.connect(self.test_combo_update)
            self.test_combo.activated.connect(self.get_and_load)

            # Result Pane]
            self.d_p = self.data_pane()
            self.g_p = self.graph_pane()
            body_layout.addLayout(self.d_p)
            # tail_layout.addLayout(self.g_p)  # no graph to display

            main_layout.addLayout(head_layout)
            main_layout.addLayout(body_layout)
            main_layout.addLayout(tail_layout)

            self.setLayout(main_layout)

        def data_pane(self):
            data_pane = QHBoxLayout()
            return data_pane

        def graph_pane(self):
            graph_pane = QVBoxLayout()
            graphs = ['1']
            for graph in graphs:
                plot = TradeHunterApp.MplCanvas()
                graph_pane.addWidget(plot)
            return graph_pane

        def back(self):
            self.close()

        # Reload data pane
        def create_labels(self):

            labels = []
            labels_2 = []
            layouts = []

            data_per_col = 20

            col_layouts = 1 + len(self.summary_dict.keys()) // data_per_col
            keys = self.summary_dict.keys()

            for i in range(col_layouts):

                left_result_body = QVBoxLayout()
                right_result_body = QVBoxLayout()

                for u in range(i * data_per_col, (i + 1) * data_per_col):

                    if len(keys) <= u:
                        break
                    key = list(keys)[u]

                    _label = QLabel(key)
                    text = self.summary_dict[key]
                    if try_float(text):
                        float_text = try_float(text)
                        _label_2 = QLabel("{:,}".format(round(float_text, 2)))
                    else:
                        _label_2 = QLabel(str(text))
                        if key == "datasets":
                            _label_2.setWordWrap(True)

                    labels.append(_label)
                    labels_2.append(_label_2)

                    left_result_body.addWidget(_label)
                    right_result_body.addWidget(_label_2)

                self.d_p.addLayout(left_result_body)
                self.d_p.addLayout(right_result_body)

                layouts.append(left_result_body)
                layouts.append(right_result_body)

            self.labels = labels
            self.labels_2 = labels_2
            self.layouts = layouts

        def delete_labels(self):
            for label in self.labels:
                label.deleteLater()
            for label in self.labels_2:
                label.deleteLater()

        # Update combo boxes
        def robot_combo_update(self):
            self.robot_combo.clear()
            robots = get_tested_robot_list()
            for robot in robots:
                self.robot_combo.addItem(robot)
            if len(robots) < 1:
                alert_w = QMessageBox('No tests found')
                alert_w.show()
            if self.robot_combo.currentText():
                self.test_combo_update()

        def test_combo_update(self):
            robot_name = self.robot_combo.currentText()
            tests = get_tests_list(robot_name)
            self.test_combo.clear()
            if len(tests) < 1:
                self.alert_window = QWidget()
                # alert_w = QMessageBox(F'No tests under {robot_name} found')
                # alert_w.show()
                alert_layout = QVBoxLayout()
                alert = QMessageBox(self.alert_window)
                alert.setText(F'No tests under {robot_name} found')
                alert_layout.addWidget(alert)
                alert.show()
            for test in tests:
                self.test_combo.addItem(test)

        # Load results

        def force_load(self, robot_name, test_name):
            self.get_and_load_name(robot_name, test_name)

            # Combo Selection to reflect this:
            self.robot_combo_update()
            idx = self.robot_combo.findText(robot_name)
            if idx == -1:
                idx == 0
            if self.robot_combo.count() < 1:
                return
            self.robot_combo.setCurrentIndex(idx)

            # Test Selection
            self.test_combo_update()
            idx = self.test_combo.findText(test_name)
            if self.test_combo.count() < 1:
                return
            self.test_combo.setCurrentIndex(idx)

        def load(self, test_result, test_meta, test_name):
            self.test_name = test_name
            self.test_result = test_result
            self.test_meta = test_meta
            self.summary_dict = self.test_result.iloc[-1]

            self.delete_labels()
            self.create_labels()

        def get_and_load(self):

            robot_name = self.robot_combo.currentText()
            test_name = self.test_combo.currentText()
            test_name = get_test_name(test_name)

            self.get_and_load_name(robot_name, test_name)

        def get_and_load_name(self, robot_name, test_name):

            self.robot_name = robot_name
            self.test_name = test_name

            test_result_name = F'{self.test_name}.csv'
            test_meta_name = F'{self.test_name}__meta.csv'

            # Load results and meta
            self.test_result = load_test_result(test_result_name, robot_name)
            self.test_meta = load_test_meta(test_meta_name, robot_name)

            self.load(self.test_result, self.test_meta, self.test_name)

    class OptimisationAnalysisPage(QWidget):  # todo
        def __init__(self, robot_name='default', test_name='default', prev_window="TradeAdvisorPage"):
            super().__init__()

        def window(self):
            pass

        def create_label(self):
            pass

        def delete_label(self):
            pass

    class VisualWindow(QWidget):
        def __init__(self, prev_window="TradeAdvisorPage"):
            self.prev_window = prev_window
            self.window()
            self.p_bar = None
            self.p_label = None
            self.main_button = None

        def window(self):
            p_layout = QVBoxLayout()

            p_bar = QProgressBar()
            p_label = QLabel('P_LABEL')

            p_layout.addWidget(p_label)
            p_layout.addWidget(p_bar)

            self.setLayout(p_layout)

    # - Plotter

    class SimplePlotter(QWidget):

        def __init__(self):
            super().__init__()
            self.window()

        def window(self):

            layout = QVBoxLayout()

            body = QVBoxLayout()

            df_layout = QHBoxLayout()
            df_select = QComboBox()
            df_select.setFixedHeight(20)
            for df_path in load_df_list():
                df_select.addItem(df_path)
            df_select.setCurrentIndex(0)

            df_layout.addWidget(QLabel('Data file:'))
            df_layout.addWidget(df_select)

            body.addLayout(df_layout)

            tail = QVBoxLayout()
            button_layout = QHBoxLayout()

            back_button = QPushButton('Back')
            test_button = QPushButton('Plot')

            def test_button_clicked():
                if not df_select.currentText():
                    self.alert_window = QWidget()
                    alert = QMessageBox(self.alert_window)
                    alert.setText('You have not selected a data file')
                    alert.show()
                else:
                    print("Select:", df_select.currentText())
                    self.plot(df_select.currentText())

            # This window does not close upon plotting.
            back_button.clicked.connect(self.back)
            test_button.clicked.connect(test_button_clicked)

            button_layout.addWidget(back_button)
            button_layout.addWidget(test_button)
            tail.addLayout(button_layout)

            layout.addLayout(body)
            layout.addLayout(tail)

            self.setLayout(layout)
            self.show()

        def back(self):
            self.close()

        def plot(self, name):
            df = load_df(name)
            # f, ax = plot_single()
            # candlestick(ax, df)

            # convert indices
            interval = get_interval(df)
            dates = df.index
            df.index = range(len(df.index))

            c = TradeHunterApp.MplCanvas(self, width=12, height=6, dpi=100)
            candlestick_plot(c.axes, df)

            # Convert axes back
            x_tick_labels = []
            for _date in dates:
                x_tick_labels.append(strtodatetime(_date).
                                     strftime(DATE_FORMAT_DICT[timedeltatoyahootimestr(interval).lower()]))
            c.axes.set(xticklabels=x_tick_labels)

            self.p_window = self.plot_window(c, F"{name}")
            self.p_window.show()

        def plot_window(self, canvas: FigureCanvasQTAgg, name: str) -> QWidget:
            p_window = QWidget()

            layout = QVBoxLayout()
            layout.addWidget(canvas)

            p_window.setLayout(layout)

            p_window.setWindowTitle(name)

            return p_window

    class SimPlotter(QWidget):

        def __init__(self):
            super().__init__()

            # Combo options
            self.robot_select = None  # robot options
            self.ivar_select = None
            self.algo_select = None  # algo
            self.algo_ivar_select = None
            self.df_select = None
            self.sim_speed = None

            # Handles for particular components
            self.plot = None  # Plot window that holds canvas
            self.canvas = None
            self.head = None
            self.body = None
            self.tail = None

            self.mode = 'Robot'  # default

            # Variables
            self.df = None
            self.xvar = {}  # XVar manual selection
            self.ivar = {}  # IVar choice
            self.svar = {}  # Sim Speed only

            self.ivar_label_dict = {}
            self.summary_dict = {}

            self.window()

        def window(self):

            layout = QVBoxLayout()

            head = QHBoxLayout()
            body = QVBoxLayout()
            tail = QVBoxLayout()
            self.head = head
            self.body = body
            self.tail = tail

            # Select Options
            df_layout = QHBoxLayout()  # Combo Boxes
            df_left_layout = QVBoxLayout()
            df_right_layout = QVBoxLayout()

            # Robot or Algo Sim
            type_combo = QComboBox()
            types = ['Robot', 'Algo']
            type_combo = QComboBox()
            test_types = ['Single', 'Multi']
            test_combo = QComboBox()

            for type in types:
                type_combo.insertItem(0, type)
            type_combo.setCurrentIndex(0)
            for type in test_types:
                test_combo.insertItem(0, type)
            test_combo.setCurrentIndex(0)

            df_label = QLabel('Data')
            df_select = QComboBox()
            robot_label = QLabel('Robot')
            robot_select = QComboBox()
            ivar_label = QLabel('IVar')
            ivar_select = QComboBox()

            df_select.setFixedHeight(20)
            robot_select.setFixedHeight(20)
            ivar_select.setFixedHeight(20)
            test_combo.setFixedHeight(20)

            sim_label = QLabel('Speed')
            sim_speed = QComboBox()  # Sim Variables

            df_left_layout.addWidget(df_label, 0.25)
            df_left_layout.addWidget(robot_label, 0.25)
            df_left_layout.addWidget(ivar_label, 0.25)
            df_left_layout.addWidget(sim_label, 0.25)
            df_right_layout.addWidget(df_select, 0.25)
            df_right_layout.addWidget(robot_select, 0.25)
            df_right_layout.addWidget(ivar_select, 0.25)
            df_right_layout.addWidget(sim_speed, 0.25)

            df_paths = load_df_list()
            df_paths.sort()
            for df_path in df_paths:
                df_select.addItem(df_path)
            for robot in load_trade_advisor_list():
                robot_select.addItem(robot)
            for speed in load_speed_suggestions():
                sim_speed.addItem(str(speed))
            df_select.setCurrentIndex(0)
            sim_speed.setCurrentIndex(0)
            robot_select.setCurrentIndex(0)

            df_layout.addLayout(df_left_layout)
            df_layout.addLayout(df_right_layout)

            self.robot_select = robot_select
            self.df_select = df_select
            self.ivar_select = ivar_select
            self.sim_speed = sim_speed

            # === IVar Layout ===
            ivar_layout = QHBoxLayout()
            ivar_left_layout = QVBoxLayout()
            ivar_right_layout = QVBoxLayout()
            self.ivar_label_dict = {}

            def load_ivar_label_list():
                robot = self.robot_select.currentText()
                ivars = load_ivar_list(robot)
                for ivar in ivars:
                    self.ivar_select.addItem(ivar)
                self.ivar_select.setCurrentIndex(0)

            def load_ivar_labels():
                ivar = self.ivar_select.currentText()
                robot = self.robot_select.currentText()
                ivar_df = load_ivar_as_dict(robot, ivar)
                self.ivar = load_ivar_as_list(robot, ivar)

                keys = self.ivar_label_dict.keys()
                for _key in ivar_df.keys():
                    if _key.lower() == 'name':
                        continue
                    if _key not in keys:
                        _label = QLabel(_key)
                        # _value = QLabel(str(ivar_df[_key].values[0]))
                        _value = QLabel(str(ivar_df[_key]))
                        self.ivar_label_dict.update({_label: _value})
                        ivar_left_layout.addWidget(_label)
                        ivar_right_layout.addWidget(_value)
                    else:
                        self.ivar_label_dict[_key].setText(ivar_df[_key])

            def load_algo_list():
                pass  # replace ivar  # todo this whole algo option in sim plotter

            def load_df():
                df_name = df_select.currentText()
                self.df = df_name

            load_ivar_label_list()
            load_ivar_labels()
            load_df()

            ivar_select.activated.connect(load_ivar_labels)
            df_select.activated.connect(load_df)

            ivar_layout.addLayout(ivar_left_layout)
            ivar_layout.addLayout(ivar_right_layout)

            # === XVar Layout ===
            xvar_layout = QHBoxLayout()  # Xvar

            xvar_left_layout = QVBoxLayout()
            xvar_right_layout = QVBoxLayout()

            lag_label = QLabel('Lag')
            lag_combo = QComboBox()
            capital_label = QLabel('Capital')
            capital_combo = QComboBox()
            capital_combo.setFixedHeight(20)
            leverage_label = QLabel('Leverage')
            leverage_combo = QComboBox()
            instrument_label = QLabel('Instrument')
            instrument_combo = QComboBox()
            type_label = QLabel('Type')  # Singular/Multi
            type_combo = QComboBox()
            commission_label = QLabel('Commission')
            commission_combo = QComboBox()
            commission_combo.setFixedHeight(20)

            test_types = ['SingleSim']
            lag_types = load_lag_suggestions()
            capital_types = load_capital_suggestions()
            leverage_types = load_leverage_suggestions()
            instrument_types = load_instrument_type_suggestions()
            commission_types = load_flat_commission_suggestions()

            # Fill in xvar combo options
            for type in test_types:
                type_combo.insertItem(0, type)
            for type in lag_types:
                lag_combo.insertItem(0, type)
            for type in capital_types:
                capital_combo.insertItem(0, str(type))
            for type in leverage_types:
                leverage_combo.insertItem(0, type)
            for type in instrument_types:
                instrument_combo.insertItem(0, type)
            for type in commission_types:
                commission_combo.insertItem(0, str(type))
            type_combo.setCurrentIndex(0)
            lag_combo.setCurrentIndex(0)
            capital_combo.setCurrentIndex(0)
            leverage_combo.setCurrentIndex(0)
            instrument_combo.setCurrentIndex(0)
            commission_combo.setCurrentIndex(0)

            xvar_left_layout.addWidget(lag_label, 1)
            xvar_left_layout.addWidget(capital_label, 1)
            xvar_left_layout.addWidget(leverage_label, 1)
            xvar_left_layout.addWidget(instrument_label, 1)
            xvar_left_layout.addWidget(type_label, 1)
            xvar_left_layout.addWidget(commission_label, 1)

            xvar_right_layout.addWidget(lag_combo, 1.5)
            xvar_right_layout.addWidget(capital_combo, 1.5)
            xvar_right_layout.addWidget(leverage_combo, 1.5)
            xvar_right_layout.addWidget(instrument_combo, 1.5)
            xvar_right_layout.addWidget(type_combo, 1.5)
            xvar_right_layout.addWidget(commission_combo, 1.5)

            xvar_layout.addLayout(xvar_left_layout, 1)
            xvar_layout.addLayout(xvar_right_layout, 1)

            # Add all col layouts
            head.addLayout(df_layout, 1)
            head.addLayout(ivar_layout, 1)
            head.addLayout(xvar_layout, 2)

            # Graph, 3 rows 1 column
            self.canvas = TradeHunterApp.MplMultiCanvas(self, 5, 4, 100, 3, 1)
            axes = self.canvas.get_axes()
            for i in range(len(axes)):
                for u in range(len(axes[i])):
                    if i != len(axes) - 1:
                        axes[i][u].get_xaxis().set_visible(False)
                    if u != 0:
                        axes[i][u].get_yaxis().set_visible(False)
            body.addWidget(self.canvas, 5)
            button_layout = QHBoxLayout()

            back_button = QPushButton('Back')
            sim_button = QPushButton('Simulate')
            stop_button = QPushButton('Stop')

            def check_input():
                if not df_select.currentText():
                    self.alert_window = QWidget()
                    alert = QMessageBox(self.alert_window)
                    alert.setText('You have not selected a data file')
                    alert.show()
                    return False
                elif not capital_combo.currentText():
                    self.alert_window = QWidget()
                    alert = QMessageBox(self.alert_window)
                    alert.setText('Please enter a valid number! e.g. 10000')
                    alert.show()
                    return False
                elif not robot_select.currentText():
                    self.alert_window = QWidget()
                    alert = QMessageBox(self.alert_window)
                    alert.setText('You have not selected a robot!')
                    alert.show()
                    return False
                return True

            def sim_button_clicked():
                # self.ivar = {}  # Saved on ivar selection
                self.xvar = {'lag': lag_combo.currentText(),
                             'capital': try_int(capital_combo.currentText()),
                             'leverage': leverage_combo.currentText(),
                             'instrument_type': instrument_combo.currentText(),
                             # test specific
                             'test_type': type_combo.currentText()}
                self.xvar = translate_xvar_dict(self.xvar)
                self.svar = {
                    'speed': try_float(sim_speed.currentText()),
                    'scope': load_setting('sim_scope')[0],  # svar:Scope
                }
                type = type_combo.currentText()
                if check_input():
                    print("Select:", df_select.currentText(), robot_select.currentText())
                    self.test(self.xvar, self.ivar, self.svar, self.robot_select.currentText(),
                              df_select.currentText(), self.canvas, type)

            # This window does not close upon plotting.
            back_button.clicked.connect(self.back)
            sim_button.clicked.connect(sim_button_clicked)

            button_layout.addWidget(back_button)
            button_layout.addWidget(sim_button)
            button_layout.addWidget(stop_button)
            tail.addLayout(button_layout)

            layout.addLayout(head, 1)
            layout.addLayout(body, 3)
            layout.addLayout(tail, 1)

            self.setLayout(layout)
            self.show()

        def test(self, xvar, ivar, svar, ta_name, df_name, type):
            data_tester = DataTester(xvar)

            if type.lower() == "robot":
                pass
            elif type.lower() == "algo":
                pass

            success, error = data_tester.simulate_single(ta_name, ivar, svar, df_name, self.canvas)

            self.plot = QWidget()
            self.plot.setWindowTitle(F'{ta_name} in {df_name}')

            if not success:
                self.alert_window = QWidget()
                alert_layout = QVBoxLayout()
                alert = QMessageBox(self.alert_window)
                alert.setText(error)
                alert_layout.addWidget(alert)
                alert.show()

        def test_outside(self, xvar, ivar, ta_name, df_name, type):
            svar = {  # Default Values
                'speed': load_setting('sim_speed')[0],
                'scope': load_setting('sim_scope')[0],
            }
            self.test(xvar, ivar, svar, ta_name, df_name, type)

        def back(self):
            self.close()

        def build_summary(self):
            # self.tail
            self.summary_layout.deleteLater()

            summary_layout = QHBoxLayout()

            keys = self.summary_dict.keys()
            n_per_col = 20

            n_col = keys // n_per_col + 1
            for i in range(n_col):

                _left = QVBoxLayout()
                _right = QVBoxLayout()

                for j in range(n_per_col):
                    k = n_per_col * i + j
                    key = keys[k]

                    _label = QLabel(key)
                    _value = QLabel(self.summary_dict[key])

                    _left.addWidget(_label)
                    _right.addWidget(_value)

                    summary_layout.addLayout(_left)
                    summary_layout.addLayout(_right)

            # Replace old summary layout
            self.tail.addLayout(summary_layout)
            self.summary_layout = summary_layout

        # Initialise options

        def build_robot_select(self):
            # Destroy algo UI
            self.algo_ivar_select.deleteLater()
            self.algo_select.deleteLater()
            self.algo_ivar_select = None
            self.algo_select = None

            self.robot_select = QComboBox()
            self.ivar_select = QComboBox()

            ta_list = load_trade_advisor_list()
            ta_list.sort(reverse=True)
            for ta in ta_list:
                self.robot_select.insertItem(0, ta)
            self.robot_select.setCurrentIndex(0)
            ivar_list = load_ivar_list(self.robot_select.currentText())
            ivar_list.sort(reverse=True)
            for ivar in ivar_list:
                self.ivar_select.insertItem(0, ivar)
            self.ivar_select.setCurrentIndex(0)

        def build_algo_select(self):
            # Destroy robot UI
            self.ivar_select.deleteLater()
            self.robot_select.deleteLater()
            self.ivar_select = None
            self.robot_select = None

            self.algo_select = QComboBox()
            self.algo_ivar_select = QComboBox()

            algo_list = load_algo_list()
            algo_list.sort(reverse=True)
            for algo in algo_list:
                self.algo_select.insertItem(0, algo)
            self.algo_select.setCurrentIndex(0)
            ivar_list = load_algo_ivar_list(self.algo_select.currentText())
            ivar_list.sort(reverse=True)
            for ivar in ivar_list:
                self.algo_ivar_select.insertItem(0, ivar)
            if len(ivar_list) > 0:
                self.algo_ivar_select.setCurrentIndex(0)

    class AlgoPlotter(QWidget):
        """Plots the sim of algorithms such as support finding, trend finding
        and exploitation algorithms."""
        pass

    # -- Utility

    class ProgressBarWindow(QWidget):
        def __init__(self):
            self.p_bar = None
            self.p_label = None
            self.window()

        def window(self):
            p_layout = QVBoxLayout()
            self.p_bar = QProgressBar()
            back_button = QPushButton('Back')
            self.p_label = QLabel('')

            back_button.clicked.connect(self.back)

            self.setLayout(p_layout)
            self.show()

        def back(self):
            self.close()

    class WarningBox(QWidget):

        def __init__(self):
            super().__init__()
            self.window()

            self.binded = None

        def window(self):
            main_layout = QVBoxLayout()

            confirm_label = QLabel('Are you sure')
            back_button = QPushButton('Ok')
            cancel_button = QPushButton('Cancel')

            back_button.clicked.connect(self.confirms)
            cancel_button.clicked.connect(self.back)

        def back(self):
            self.close()

        def confirm(self):
            self.binded()

        def bind_method(self, f):
            self.binded = f

    # -- Canvas
    class MplCanvas(FigureCanvasQTAgg):

        def __init__(self, parent=None, width=5, height=4, dpi=100):
            fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = fig.add_subplot(111)
            super(TradeHunterApp.MplCanvas, self).__init__(fig)

        def get_axes(self):
            return self.axes

    class MplMultiCanvas(FigureCanvasQTAgg):

        def __init__(self, parent=None, width=5, height=4, dpi=100, rows=1, cols=1):
            fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = []
            height_ratios = [rows / 2.0 + 1]
            for i in range(1, rows):
                height_ratios.append(1)
            gs = gridspec.GridSpec(rows, cols, height_ratios=height_ratios, figure=fig)
            for i in range(rows):
                _axes = []
                for j in range(cols):
                    _axes.append(fig.add_subplot(gs[i, j]))
                self.axes.append(_axes)
            super(TradeHunterApp.MplMultiCanvas, self).__init__(fig)

        def get_axes(self):
            return self.axes

        def get_ax(self, row, col):
            return self.axes[row][col]

    class MplSubCanvas(FigureCanvasQTAgg):

        def __init__(self, parent, width, height, dpi, rows, cols):
            pass

        def get_axes(self):
            return self.axes

# ROBOT PAGE - DELETE IVARS
# RESULT PAGE - DELETE TEST RESULTS OR OPTIM RESULTS
# REDOWNLOAD ALL - DELETE ALL DATA, REDOWNLOAD ALL

# Future method: Redownload All
# Find all meta information of a data
# download everything past latest data point,
# if its too far back, delete file, redownload
# Unmentioned files, delete all.
# 3 categories: unmentioned, new, mentioned
# Delete unmentioned, download new, special care for mentioned
