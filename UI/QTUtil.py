import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal, pyqtBoundSignal
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QPlainTextEdit, QSlider, QWidget, QVBoxLayout, QLabel, \
    QHBoxLayout, QPushButton, QGroupBox
from util.langUtil import check_if_valid_timestr
from util.mathUtil import try_float, try_int


def get_datatable_sheet(table: QTableWidget):
    map = []
    for i in range(table.rowCount()):
        row = []
        for u in range(table.columnCount()):
            if table.item(i, u):
                row.append(table.item(i, u).text())
            else:
                row.append("")

        map.append(row)

    # Ignore not-full rows AND symbol/interval not allowed!
    map = [row for row in map if row[0] and row[1] and row[2]
           and check_if_valid_timestr(row[1]) and check_if_valid_timestr(row[2])]
    data = {
        'symbol': [row[0] for row in map],
        'interval': [row[1] for row in map],
        'period': [row[2] for row in map],
    }
    df = pd.DataFrame(data)

    return df


def get_datatable_sheet_all(table: QTableWidget):
    map = []
    for i in range(table.rowCount()):
        row = []
        for u in range(table.columnCount()):
            if table.item(i, u):
                row.append(table.item(i, u).text())
            else:
                row.append("")

        map.append(row)

    map = [row for row in map if row[0] or row[1] or row[2]]

    data = {
        'symbol': [row[0] for row in map],
        'interval': [row[1] for row in map],
        'period': [row[2] for row in map],
    }
    df = pd.DataFrame(data)

    return df


def get_datatable_sheet_col(table: QTableWidget, col: int):
    map = []
    for i in range(table.rowCount()):
        map.append(table.item(i, col).text())
        # Stop at a blank
        if not map[-1]:
            break
    map = [row for row in map if row]

    data = {
        'data': [row[0] for row in map],
    }
    df = pd.DataFrame(data)

    return df


def set_datatable_sheet(table: QTableWidget, dataset_df: pd.DataFrame):
    clear_table(table)

    columns = dataset_df.columns

    for i in range(len(dataset_df)):  # rows
        for u in range(len(columns)):
            data_item = QTableWidgetItem(dataset_df[columns[u]][i])
            table.setItem(i, u, data_item)


def set_dataset_table(table: QTableWidget, ds_names):
    for i in range(len(ds_names)):
        data_item = QTableWidgetItem(ds_names[i])
        table.setItem(i, 0, data_item)


def get_dataset_table(table: QTableWidget):
    ds_names = []
    for i in range(table.rowCount()):
        if table.item(i, 0):
            ds_names.append(table.item(i, 0).text())
    ds_names = [ds_name for ds_name in ds_names if ds_name]

    return ds_names


def set_cell_sheet(table: QTableWidget, text: str, row: int, col: int):
    data_item = QTableWidgetItem(text)
    table.setItem(row, col, data_item)


def set_col_cell_sheet(table: QTableWidget, text: str, col: int):
    data_item = QTableWidgetItem(text)
    row_count = len(get_datatable_sheet_col(table, col))
    print(F'fitting {text} in {table.rowCount()}, {col} of {table}')
    table.setItem(row_count, col, data_item)


def clear_table(table: QTableWidget):
    for i in range(table.rowCount()):
        for u in range(table.columnCount()):
            item = QTableWidgetItem("")
            table.setItem(i, u, item)


def count_table(table: QTableWidget):
    rows = 0
    for i in range(table.rowCount()):
        empty = False
        for u in range(table.columnCount()):
            if not table.item(i, u):
                empty = True
        if not empty:
            rows += 1
    return rows


def full_only(map):
    _map = []
    for i in range(map.count()):
        row = []
        to_row = True
        for u in range(map.count()):
            if map[i][u]:
                row.append(u)
            else:
                to_row = False
                break
        if to_row: _map.append(row)

    return _map


# util GUI classes

ALPHABETS = (Qt.Key_A, Qt.Key_B, Qt.Key_C, Qt.Key_D, Qt.Key_E, Qt.Key_F, Qt.Key_G, Qt.Key_H, Qt.Key_I
             , Qt.Key_J, Qt.Key_K, Qt.Key_L, Qt.Key_M, Qt.Key_N, Qt.Key_O, Qt.Key_Q, Qt.Key_R, Qt.Key_S
             , Qt.Key_T, Qt.Key_U, Qt.Key_V, Qt.Key_W, Qt.Key_X, Qt.Key_Y, Qt.Key_Z,)
NUMERALS = (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9, Qt.Key_0,)
ALPHANUMERICS = ALPHABETS + NUMERALS
SPACE_KEYS = (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Tab,)
SPECIAL_CHARS = (Qt.Key_Exclam, Qt.Key_At, Qt.Key_NumberSign, Qt.Key_Dollar)  # !@#$%^&*()-_=+[{]};:'",<.>/?\|`~


class PlainTextEdit(QPlainTextEdit):
    def keyPressEvent(self, event):
        if event.key() not in ALPHANUMERICS:
            return
        super().keyPressEvent(event)


class NumericTextEdit(QPlainTextEdit):
    def keyPressEvent(self, event):
        if event.key() in SPACE_KEYS:
            return
        if event.key() in (Qt.Key_A, Qt.Key_B, Qt.Key_C, Qt.Key_D, Qt.Key_E, Qt.Key_F, Qt.Key_G, Qt.Key_H, Qt.Key_I
                           , Qt.Key_J, Qt.Key_K, Qt.Key_L, Qt.Key_M, Qt.Key_N, Qt.Key_O, Qt.Key_Q, Qt.Key_R, Qt.Key_S
                           , Qt.Key_T, Qt.Key_U, Qt.Key_V, Qt.Key_W, Qt.Key_X, Qt.Key_Y, Qt.Key_Z,):
            return
        super().keyPressEvent(event)


class TextEditForm(QPlainTextEdit):
    """Pressing 'tab' moves to the next input"""
    pass


class DiscreteSlider(QSlider):
    pass


class DoubleSlider(QSlider):
    doubleValueChanged = pyqtSignal(float)
    # doubleValueChanged = pyqtBoundSignal(float)

    def __init__(self, *args, **kwargs):
        super(DoubleSlider, self).__init__(*args, **kwargs)
        self._min = 0
        self._max = 99
        self.interval = 1
        # super().valueChanged()
        super().valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        self.doubleValueChanged.emit(self.value())

    def setValue(self, value):
        index = round((value - self._min) / self.interval)
        return super(DoubleSlider, self).setValue(index)

    def value(self):
        return self.index * self.interval + self._min

    @property
    def index(self):
        return super(DoubleSlider, self).value()

    def setIndex(self, index):
        return super(DoubleSlider, self).setValue(index)

    def setMinimum(self, value):
        self._min = value
        self._range_adjusted()

    def setMaximum(self, value):
        self._max = value
        self._range_adjusted()

    def setInterval(self, value):
        # To avoid division by zero
        if not value:
            raise ValueError('Interval of zero specified')
        self.interval = value
        self._range_adjusted()

    def _range_adjusted(self):
        number_of_steps = int((self._max - self._min) / self.interval)
        super(DoubleSlider, self).setMaximum(number_of_steps)


# class DoubleStepSlider(QSlider):
#     doubleValueChanged = pyqtSignal(float)
#
#     def __init__(self, *args, **kargs):
#         super(DoubleStepSlider, self).__init__(*args, **kargs)
#         self._min = 0
#         self._max = 99
#         self.interval = 1
#
#         self.steps = 100
#
#     def emitDoubleValueChanged(self):
#         self.doubleValueChanged.emit(self.value)
#
#     def setSteps(self, steps):
#         self.steps = steps
#
#     def setInterval(self, value):
#         self.value = value


class SliderText(QWidget):

    def __init__(self, _min=0, _max=1, orientation=Qt.Horizontal):
        super().__init__()

        self.slider = None
        self.text = None
        self.main_layout = None
        # self.box = None
        self.box_layout = None
        self.max = _max
        self.min = _min
        self.orientation = orientation

        self.edited = False

        self.window()

    def window(self):

        self.slider = DoubleSlider(self.orientation)
        self.text = NumericTextEdit()

        self.main_layout = QHBoxLayout()

        self.main_layout.addWidget(self.slider)
        self.main_layout.addWidget(self.text)

        # Group boxing
        # self.box = QGroupBox()
        # self.box.setLayout(self.main_layout)
        # self.box_layout = QHBoxLayout()
        # self.box_layout.addWidget(self.box)

        # Connect events, set style
        self.slider.doubleValueChanged.connect(self.sliderUpdate)  # double value changed error happens here
        # self.slider.valueChanged.connect(self.sliderUpdate)
        self.text.textChanged.connect(self.textUpdate)

        self.slider.setMaximum(self.max)
        self.slider.setMinimum(self.min)

        # init values
        self.slider.setValue(self.min)
        self.text.setPlainText(str(self.min))

        # self.setLayout(self.box_layout)
        self.setLayout(self.main_layout)
        self.setWindowTitle('Testing here!')
        self.show()

    # == Events ==

    def sliderUpdate(self):
        e = self.text.document().toPlainText()
        v = self.slider.value()
        print(F"(Slider) Compare between {e} and {v}")
        if e != str(v):
            self.edited = False
        if not self.edited:
            print(F"Setting text to slide value {str(self.slider.value())}")
            self.text.setPlainText(str(self.slider.value()))
            self.edited = True

    def textUpdate(self):
        e = self.text.document().toPlainText()
        v = self.slider.value()
        print(F"(Text) Compare between {e} and {v}")
        if e != str(v):
            self.edited = False

        if not self.edited:
            # Check if text is a number
            e = self.text.document().toPlainText()
            if not e == '0' and not try_float(e):
                e = 0
            # Check if within bounds
            v = try_float(e)
            if v < self.min:
                v = self.min
            if v > self.max:
                v = self.max
            print(F"Setting slider to {v}")
            self.slider.setValue(v)
            # self.text.setPlainText(str(v)) # infinite loop
            self.edited = True
            # check if same, if yes, self.edited = False

    # == Default ==

    def back(self):
        self.close()

    def setMinimum(self, value):
        # if min > self.slider.maximum:
        #     self.slider.setMinimum(self.slider.Maximum)
        self.slider.setMinimum(value)
        self.min = value
        if self.slider.value() < self.min:
            self.slider.setValue(self.min)

    def setMaximum(self, value):
        self.slider.setMaximum(value)
        self.max = value
        if self.slider.value() > self.max:
            self.slider.setValue(self.max)

    def setValue(self, val):
        print("Setting all values to " + str(val))
        self.slider.setValue(val)
        self.text.setPlainText(str(val))  # weird this one only activates for variability constant

    def getValue(self):
        return self.slider.value()

    def tickPosition(self):
        return self.slider.tickPOsition()

    def setFixedHeight(self, h: int) -> None:
        self.slider.setFixedHeight(h)
        self.text.setFixedHeight(h)


class DiscreteSliderText(QWidget):

    def __init__(self, _min=0, _max=1, orientation=Qt.Horizontal):
        super().__init__()

        self.slider = DiscreteSlider(orientation)
        self.text = NumericTextEdit()

        self.main_layout = QHBoxLayout()

        self.main_layout.addWidget(self.slider)
        self.main_layout.addWidget(self.text)

        # Connect events, set style
        self.slider.valueChanged.connect(self.sliderUpdate)
        self.text.textChanged.connect(self.textUpdate)

        self.orientation = orientation
        self.max = _max
        self.min = _min
        self.slider.setMaximum(_max)
        self.slider.setMinimum(_min)

        self.setLayout(self.main_layout)

    # == Default ==

    def back(self):
        self.close()

    def setMinimum(self, min):
        # if min > self.slider.maximum:
        #     self.slider.setMinimum(self.slider.Maximum)
        self.slider.setMinimum(min)
        self.min = min
        if self.slider.value() < self.min:
            self.slider.setValue(self.min)

    def setMaximum(self, max):
        self.slider.setMaximum(max)
        self.max = max
        if self.slider.value() > self.max:
            self.slider.setValue(self.max)

    def setValue(self, val):
        self.slider.setValue(val)
        self.text.setPlainText(str(val))

    def getValue(self):
        return self.slider.value()

    def tickPosition(self):
        return self.slider.tickPosition()

    def setInterval(self, val):
        # To avoid division by zero
        self.slider.setTickInterval(val)
        self.slider.setSingleStep(val)  # arrow-key step-size
        self.slider.setPageStep(val)  # mouse-wheel/page-key step-size

    # == Events ==

    def sliderUpdate(self):
        e = self.text.document().toPlainText()
        v = self.slider.value()
        if e != str(v):
            self.edited = False
        if not self.edited:
            self.text.setPlainText(str(self.slider.value()))
            self.edited = True

    def textUpdate(self):
        e = self.text.document().toPlainText()
        v = self.slider.value()
        if e != str(v):
            self.edited = False

        if not self.edited:
            # Check if text is a number
            e = self.text.document().toPlainText()
            if not e == '0' and not try_float(e):
                e = 0
            # Check if within bounds
            v = try_float(e)
            if v < self.min:
                v = self.min
            if v > self.max:
                v = self.max
            self.slider.setValue(int(v))  # which will inadvertently change this again
            # self.text.setPlainText(str(v)) # infinite loop
            self.edited = True
            # check if same, if yes, self.edited = False

    def setFixedHeight(self, h: int) -> None:
        self.slider.setFixedHeight(h)
        self.text.setFixedHeight(h)


class TwinLabel(QWidget):
    """Test widget for testing multi-labels and multi-widgets in general"""

    def __init__(self):
        super().__init__()
        self.window()

    def window(self):
        hbox = QHBoxLayout()
        label1 = QLabel('Label1')
        label2 = QLabel('Label2')
        hbox.addWidget(label1)
        hbox.addWidget(label2)
        self.setLayout(hbox)

        self.show()


# Window


class ConfirmWindow(QWidget):
    def __init__(self, text='Are you sure?', title='Confirmation Window',
                 button_text_1='Confirm', button_text_2='Cancel'):
        super().__init__()
        # UI Widgets' text
        self.text = text
        self.title = title
        self.button_text_1 = button_text_1
        self.button_text_2 = button_text_2
        # Button binded function
        self.f = None
        # Main
        self.window()

    def window(self):
        layout = QVBoxLayout()
        label = QLabel(self.text)
        layout.addWidget(label)
        self.setWindowTitle(self.title)

        button_layout = QHBoxLayout()
        confirm_button = QPushButton(self.button_text_1)
        cancel_button = QPushButton(self.button_text_2)
        button_layout.addWidget(confirm_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        cancel_button.clicked.connect(self.back)
        confirm_button.clicked.connect(self.confirm)

        self.setLayout(layout)

    def bind_function(self, f):
        self.f = f

    def confirm(self):
        self.f()

    def back(self):
        self.close()


class QuickAlertWindow(QWidget):
    # todo on "enter" close window! (future?)

    def __init__(self, text="Alert!", title="Alert Window", button_text="OK"):
        super().__init__()
        # UI Widgets' text
        self.text = text
        self.title = title
        self.button_text = button_text
        # Main
        self.window()

    def window(self):
        layout = QVBoxLayout()
        label = QLabel(self.text)
        layout.addWidget(label)
        self.setWindowTitle(self.title)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_button = QPushButton(self.button_text)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        cancel_button.clicked.connect(self.back)

        self.setLayout(layout)

    def back(self):
        self.close()
