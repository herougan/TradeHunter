'''FMACD Robot'''
from robot.abstract.robot import robot
from util.dataTestingUtil import create_profit_df_from_list, create_signal_df_from_list, create_summary_df_from_list
from datetime import datetime


class FMACDRobot(robot):
    IVAR_STEP = 0.05

    N_ARGS = 2
    ARGS_STR = ['stop_loss', 'take_profit']
    ARGS_DEFAULT = [1, 1.5]
    ARGS_RANGE = [[0.01, 10], [0.01, 10]]

    def __init__(self, ivar=ARGS_DEFAULT, xvar={}):
        if len(ivar) == 2:
            self.ivar = ivar
        else:
            self.ivar = FMACDRobot.ARGS_DEFAULT
        self.ivar_range = FMACDRobot.ARGS_RANGE

        # External attributes
        self.xvar = xvar

        # Indicator Data
        self.indicators = {
            'SMA200': 0,
            'EMA': 0,
            'MACD1': 0,
            'MACD2': 0,
            'MACDHIST': 0,
        }

        # Statistical Data
        self.profit_data = []
        self.asset_data = []
        # Operational variables
        self.signals = []
        self.signal_headers = []
        self.completed_signals = []

        # Data
        self.df = {}

        # Meta does not matter, but just in case
        self.symbol = ""

        # Upon completion
        self.profit_df = None
        self.signal_df = None
        self.summary_df = None

        # Status
        self.market_active = False
        self.started = False
        self.last_date = datetime.now()

    def start(self):
        self.reset()

    def open(self):
        return

    def get_ivar_len(self):
        return 0

    def ivar(self, *ivar):
        self.ivar = ivar

    def mutate_ivar(self, idx, _ivar):
        self.ivar[idx] = _ivar

    def step_ivar(self, idx, up=True):
        if len(self.ivar_range) >= idx:
            i = -1
            if up:
                i = 1
            self.ivar[idx] += (self.ivar_range[idx][1] - self.ivar[idx]) * robot.IVAR_STEP * i

    def step(self, tick):
        """Tick is this tick's data."""
        df = []
        index = range(len(df))

        profit = 0
        liquid_assets = 0
        self.profit_data.append(profit)
        self.asset_data.append(liquid_assets)

    def sell(self, ind):
        # self.signals = [signal for signal in self.signals if signal['type'] == 'Buy']
        signal = self.signals[ind]
        del self.signals[ind]
        completed_signal = {

        }
        self.completed_signals.append(completed_signal)

    def reset(self):
        self.steps = 0

        self.profit_df = None
        self.signal_df = None
        self.summary_df = None

    def on_complete(self):
        self.profit_df = create_profit_df_from_list(self.profit_data, self.asset_data)
        self.signal_df = create_signal_df_from_list(self.completed_signals, self.signals)
        self.summary_df = create_summary_df_from_list(self.profit_data, self.asset_data, self.completed_signals)

    # START

    def start_test(self, data_meta: List[str], interval: timedelta):
        """Begin by understanding the incoming data.
        Setup data will be sent
        E.g. If SMA-200 is needed, at the minimum, the past 400 data points should be known.

        old_data = retrieve()

        From then on, the robot receives data realtime - simulated by feeding point by point. (candlestick)
        """
        self.interval = interval
        p_df = self.retrieve_prepare()

        self.build_indicators()

        # Set up Indicators

        self.started = True

    def retrieve_prepare(self):

        df = retrieve("symbol", datetime.now(), datetime.now() - self.interval * self.prepare_period, self.interval, False, False)


        # If no such file, SMA will be empty.

        df = {}

        return df

    def next(self, candlestick: pd.DataFrame):

        # update indicators

        # check signals

        # make signals

        # end-
        pass

    def next(self, candlestick_list: pd.DataFrame):
        """Same as above, but only when updates are missed so the whole backlog would
         included. Only candlestick_list[-1] will be measured for signals. The time has set
         for the rest but they are needed to calculate the indicators."""
        pass

    def finish(self):
        pass

    # De-init

    def get_data(self):
        return self.df

    # Indicators

    def build_indicators(self):
        pass

    def indicator_next(self, candlestick: pd.DataFrame):
        pass

    def get_indicator_df(self):
        pass

    # Signals

    def get_signal_df(self):
        pass

    def get_pl_df(self):
        pass

    def record_profits(self):
        pass

    def record_signals(self):
        pass

    # utility

    def add_signal(self):
        # if signals were = { 'date': [], 'vol': [] }
        # checking for a signal would be troublesome!
        # more convenient if for signal in signals!
        # therefore signals = [{ 'date': date, 'vol': vol, ...}...]
        pass

    def check_stop_loss(self):
        pass

    def check_profit_loss(self):
        pass

    def close_trade(self):

        # Calculate profit
        pass
