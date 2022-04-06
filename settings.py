import os

DATASET_CHANGES_PATH = F''
COMMON_SYMBOLS_PATH = F''
COMMON_INTERVALS_PATH = F''
COMMON_PERIODS_PATH = F''

DATASETDEF_FOLDER = F''
DATA_FOLDER = F''
RESULTS_FOLDER = F''
TESTS_FOLDER = F''
ROBOTS_FOLDER = F''
BUILDER_FOLDER = F''

EVALUATION_FOLDER = F'static/results/evaluation'
OPTIMISATION_FOLDER = F'static/results/optimisation'

ROOT = F''
STATIC = F'{os.getcwd()}'

DATATABLE = {
    'length': [100]
}

SUGGESTIONS = {
    'symbols': [
                '--stock--',
                'AAPL', 'TLSA', 'GOOGL', 'INTC', 'MSFT', 'AMZN', 'ROKU', 'HD', 'BAC', 'PLTR', 'JPM', 'RBLX',
                '--forex--',
                'USD-CAD', 'GBP-USD', 'AUD-USD', 'NZD-USD', 'JPY-USD', 'USD-JPY',
                '--common--',
                'OIL', 'COTTON', 'COFFEE', 'SUGAR', 'WHEAT', 'SOY',
                '--metals--',
                'GOLD', 'SILVER', 'PLATINUM', 'PALLADIUM', 'COPPER',
                '--indexes--',
                '^STI', '^N225', '0000001.SS', '^GSPC', '^DJI', '^IXIC', '^BSESN', '^HSI',
                '--crypto--',
                'BTC-USD', 'ETH-USD', 'BNB-USD', 'DOGE-USD', 'USDT-USD',
                ],
    'intervals': ['1M', '2M', '5M', '15M', '30M', '60M', '1h', '90M', '1d', '5d', '1wk', '2wk', '1m', '3m', ],
    'periods': ['1d', '5d', '1wk', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max', ],
    'lag': ['0 ms', '10 ms', '100 ms', '1 s', '10 s', ],
    'leverage': ['10:1', '1:1', '1:10', '1:100', '1:200', '1:500', '1:1000', ],
    'instrument_type': ['Forex', '0.01 Forex', 'Crypto', 'Stock', 'Index', 'Futures', 'Hedge'],
    'sim_speed': ['0.5x', '1x', '2x', '5x', '10x', '25x', '100x', '500x', '1000x'],
    'capital': [1000, 10000, 100000, 1000000,],
    'flat_commission': [0, 0.01, 0.1, 1,],
    'contract_size': [100000, 10000, 1000],  # Set in stone
    'simulation': {
        'speed': [0.5, 1, 2, 4, 8, 16]
    }
}
PLOTTING_SETTINGS = {
    'bar_width_to_interval': {
        '60m': 1,
        '90m': 1.2,
        '1d': 1.6,
        '1wk': 2.1,
    },
    'date_format_to_interval': {
        '60m': '%Y-%m-%d %H:%M',
        '90m': '%Y-%m-%d %H:%M',
        '1d': '%Y-%m-%d',
        '1wk': '%Y-%m-%d',
    },
    'candle_width_to_scope': {
        15: 1,
        40: 2,  # 15 - 40 etc.
        100: 5,
        250: 12,
        -1: 30,  # 250 - ...
    },
    'candle_fixed_width': 8,
    'candle_ratio': 6,  # Thin part of candle = candle_width / candle_ratio
    'plot_margin': [0.1, 0.1],
    'graph_margin': [0.2, 0.2, 0.25],  # x, y, z (if appl.); inner plot margin
}
