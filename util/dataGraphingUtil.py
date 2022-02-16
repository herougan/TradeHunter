def candlestick(ax, df):
    """Plots candlestick data based on df on ax"""

    pass


def sma(ax, df, period):
    pass


def ema(ax, df, period, exp):
    pass


def profit_graph(ax, df, ls):
    pass


def load_signals(ax, sdf):
    """Loads buy signatures onto time series"""
    for datum in sdf:
        id = datum['id']
        datum_close = next(d for d in sdf if d['id'] == id and d['date'] != datum['date'])
        # Remove datums
        if not datum_close:
            sdf.drop([id])
        else:
            sdf.drop[id, datum_close['id']]


        pass
    # Number of "error" keys

    # Fetch successful shorts

    # Successful longs

    # Unsuccessful longs and shorts

    pass


def mac_diagram(ax, macd_df):

    pass
