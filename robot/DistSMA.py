"""DistSMA Robot"""

from robot.abstract.robot import robot

class DistSMA(robot):
    """A robot that trades when the current price is heavily against the current trend.
    SMAs of different periods are employed for: Measuring trend or variability and determining
    pullback chance. Each SMA gives a score and the sum of these are measured against the criterion
    """

    ARGS_DICT = {
        'SMA_period_1': {},
        'SMA_period_2': {},
        'SMA_period_3': {},
        'SMA_period_4': {},
        'SMA_period_5': {},
        #
        'SMA_score_1': {},
        'SMA_score_2': {},
        'SMA_score_3': {},
        'SMA_score_4': {},
        'SMA_score_5': {},
        #
        'prepare_period': {},
    }
    PREPARE_PERIOD = 20  # replace in __init__