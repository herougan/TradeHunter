# Filtered MACD Robot
'''Filtered MACD Robot'''
from robot.abstract.robot import robot

# Static Variables:
N_ARGS = 2
ARGS_STR = ['stop_loss', 'take_profit']
ARGS_DEFAULT = [1, 1.5]
# todo move into the robot!

class FilterRobot(robot):

    def __init__(self, r3):
        self.r3 = r3

    def open(self):
        return

    def get_ivar_len(self):
        return 0
