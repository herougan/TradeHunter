import pandas as pd
from PyQt5.QtWidgets import QTableWidget


def get_sheet(table: QTableWidget):
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


def set_sheet(table: QTableWidget, dataset_df: pd.DataFrame):

    # Get rows
    print(dataset_df.columns)

    for i in range(dataset_df.count()):
        for u in range(dataset_df.count()):
            for index, row in dataset_df.iterrows():
                print(row['c1'], row['c2'])
    pass


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