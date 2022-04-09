# Cost is in terms of millions.
from random import random

STARFORCE_TARGET = 17
STARFORCE_START = 0

EQUIPMENT_LEVEL = 200
REPLACEMENT_COST = 150

# Static Info

# STARFORCE_COST = []  1000 + LVL ^ 3 * (C + 1)/25
def starforce_cost(level, starforce):
    a, b = 0, 0
    if starforce < 10:
        a, b = 1, 25
    elif starforce < 15:
        a, b = 2.7, 400
    elif starforce < 25:
        a, b = 2.7, 200
    return 1000 * pow(level, 3) + pow((starforce + 1), a) / b
BOOM_CHANCE = [0, 0, 0, 0, 0,
               0, 0, 0, 0, 0,
               0, 0, 0.06, 0.013, 0.014,
               0.021, 0.021, 0.021, 0.028, 0.028,
               0.07, 0.07, 0.194, 0.294, 0.396]
REDUCTION = [0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0.55, 0.594, 0.637, 0.686,
             0, 67.9, 67.9, 67.2, 67.2,
             0, 63, 77.6, 68.6, 59.4]
FAILURE = [0.05, 0.1, 0.15, 0.15, 0.2,
           0.25, 0.3, 0.35, 0.4, 0.45,
           0.5, 0, 0, 0, 0,
           0, 0, 0, 0, 0,
           0, 0, 0, 0, 0]
SUCCESS = [0.95, 0.90, 0.85, 0.85, 0.8,
           0.75, 0.7, 0.75, 0.6, 0.55,
           0.5, 0.45, 0.4, 0.35, 0.3,
           0.3, 0.3, 0.3, 0.3, 0.3,
           0.3, 0.3, 0.03, 0.02, 0.01]
SAFEGUARD_COST = 2  # 2 times
STARFORCE_RESET = 11
ARCANE_FORCE = 0
AUTHENTIC_FORCE = 0
DARK_STARFORCE = 0

# Events
FIVE_TEN_FIFTEEN_EVENT = False
THIRTY_PERCENT = False
NO_BOOM_UNDER_FIFTEEN = False
DOUBLE_ENHANCE_UNDER_TEN = False
TWELVE_FIFTEEN_NO_BOOM = False

STAR_CATCHING = False
STAR_CATCHING_CHANCE = 0.9  # Fail to catch 10% of the time, (*104.5% chance)

SAFEGUARD = True

MVP = 0  # 0.03 for 3% discount. Silver/Gold/Diamond: 3/5/10%

# Methods
SAFEGUARD_ALL = True  # If False, Safeguard is never turned on
REPLACE_ON_BOOM = True


# Run type
EACH_LEVEL_BEST_FIND = True
FULL_RUN_CALC = False

RUNS = 1000

total_full_runs = 0

if FULL_RUN_CALC:
    while total_full_runs < RUNS:
        total_cost = 0
        total_attempts = 0
        total_booms = 0

        starforce = STARFORCE_START

        get_out = 0
        while starforce < STARFORCE_TARGET and get_out < 100000:

            # Attempt starforce
            r = random.random()
            pass
            get_out += 1
elif EACH_LEVEL_BEST_FIND:
    pass
