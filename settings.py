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
                'USDCAD', 'GBPUSD', 'AUDUSD', 'NZDUSD',
                '--common--',
                'OIL', 'COTTON', 'COFFEE', 'SUGAR', 'WHEAT', 'SOY',
                '--metals--',
                'GOLD', 'SILVER', 'PLATINUM', 'PALLADIUM', 'COPPER',
                ],
    'intervals': ['1M', '2M', '5M', '15M', '30M', '60M', '1h', '90M', '1d', '5d', '1wk', '2wk', '1m', '3m', ],
    'periods': ['1wk', '1m', '3m', '6m', '1y', '2y', '5y', 'max', ],
}
