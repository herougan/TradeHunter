# Filtered MACD Robot
'''Filtered MACD Robot'''
from robot.abstract.robot import robot

# Static Variables:
from util.dataTestingUtil import create_profit_df_from_list, create_signal_df_from_list, create_summary_df_from_list

N_ARGS = 2
ARGS_STR = ['stop_loss', 'take_profit']
ARGS_DEFAULT = [1, 1.5]


class FMACDRobot(robot):

    def __init__(self, ivar=ARGS_DEFAULT):
        if len(ivar) == 2:
            self.ivar = ivar
        else:
            self.ivar = ARGS_DEFAULT

        self.profit_data = []
        self.asset_data = []

        self.signals = []  # todo ask what MT5 calls these
        self.signal_headers = []
        self.completed_signals = []

        # Upon completion

        self.profit_df
        self.signal_df
        self.summary_df

    def start(self):
        self.reset()


    def open(self):
        return

    def get_ivar_len(self):
        return 0

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

