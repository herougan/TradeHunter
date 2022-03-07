import pandas as pd
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem


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

    data = {
        'symbol': [row[0] for row in map if row[0] and row[1] and row[2]],
        'interval': [row[1] for row in map if row[0] and row[1] and row[2]],
        'period': [row[2] for row in map if row[0] and row[1] and row[2]],
            }
    df = pd.DataFrame(data)

    return df


def set_datatable_sheet(table: QTableWidget, dataset_df: pd.DataFrame):
    clear_table(table)

    columns = dataset_df.columns

    for i in range(len(dataset_df.index)):  # rows
        for u in range(len(columns) - 1):
            data_item = QTableWidgetItem(dataset_df[columns[u+1]][i])
            table.setItem(i, u, data_item)


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


def set_test_result(table: QTableWidget, result_df: pd.DataFrame, meta_df: pd.DataFrame):
    pass # todo