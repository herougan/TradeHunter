import os

DATASET_CHANGES_PATH = F'static/common/datasetchanges.txt'
COMMON_SYMBOLS_PATH = F'static/common/datasetchanges.txt'
COMMON_INTERVALS_PATH = F'static/common/datasetchanges.txt'
COMMON_PERIODS_PATH = F'static/common/datasetchanges.txt'

DATASETDEF_FOLDER = F'static/datasetdef'
DATA_FOLDER = F'static/data'
RESULTS_FOLDER = F''
TESTS_FOLDER = F''
BUILDER_FOLDER = F''

EVALUATION_FOLDER = F'static/results/evaluation'
OPTIMISATION_FOLDER = F'static/results/optimisation'
ALGO_ANALYSIS_FOLDER = F'static/results/algo'

ROBOT_FOLDER = F'robot'
ALGO_FOLDER = F'algo'
IVAR_SUB_FOLDER = F'/ivar'

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
        'CAD=X', 'EURUSD=X', 'CHF=X', 'GBPUSD=X', 'NZDUSD=X', 'AUDUSD=X',
        'JPY=X', 'SGD=X', 'CNY=X', 'ZAR=X', 'INR=X',
        'EURCAD=X', 'EURAUD=X', 'EURJPY=X', 'EURCHF=X', 'EURGBP=X', 'AUDCAD=X',
        'GBPCHF=X', 'GBPJPY=X', 'CHFJPY=X', 'AUDJPY=X', 'AUDNZD=X',
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
    'instrument_type': ['Crypto', 'Stock', 'Index', 'Futures', 'Hedge', 'Forex'],
    'sim_speed': ['0.5x', '1x', '2x', '5x', '10x', '25x', '100x', '500x', '1000x'],
    'capital': [1000, 10000, 100000, 1000000, ],
    'flat_commission': [0, 0.01, 0.1, 1, ],
    'contract_size': [100000, 10000, 1000],  # Set in stone
    'simulation': {
        'speed': [0.5, 1, 2, 4, 8, 16]
    },
    # Optimisation
    'optim_width': [3, 10, 100, 300, 1000, 5000],
    'optim_depth': [3, 10, 25, 100, 300, 700, 1000],
    # Simulation
    'sim_speed': [1],
    'sim_scope': [80, 100],
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
TESTING_SETTINGS = {
    'test_runs': 3
}
OPTIMISATION_SETTINGS = {
    'max_runs': 100,  # optim_depth overwrites this
    'arg_step_size': 5,  # Adjusts step size during optimisation. (Default: 5)
    # 0.05 = 5% of minimum step value (if appl.) or 5%/10 of full range - max 10%.
    # 5 = 500% of minimum step value. e.g. total percentage magnitude of 'change' vector
    # Many vector components would be <= min step value.
    'approach_step_size': 3,
    'optimisation_width': 5,
    'optimisation_depth': 3,
    'learning_rate': 1,
}


# Enum

class IVarType:
    NONE = 0
    CONTINUOUS = 1
    DISCRETE = 2  # Integer type is discrete but with step_size 1
    ENUM = 3
    TEXT = 4  # Single value, cannot be optimised
    ARRAY = 5  # Enum but for numbers only
    SEQUENCE = 6  # Same as array but implies trend/pattern while traversing array
    # Future
    VECTOR = 7  # Vector of any size
    SQ_MATRIX = 8  # Matrix of square size
    PERM = 9  # Permutation of range's values

    INTEGER = 92


class InputUIType:
    TEXT = 1
    COMBO = 2
    LIST = 3
    TABLE = 4
    NUMBER = 5
    COMBO_NUMBER = 6
    SLIDER = 7
