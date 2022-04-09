import pandas as pd
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from util.langUtil import check_if_valid_timestr


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