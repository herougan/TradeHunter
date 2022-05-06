names = {
}
creatures = {
    'bear': {
        'health': 500,
        'defence': 2,
        'attack': 30,
    },
    'deer': {
        'health': 150,
        'defence': 1,
        'attack': 30,
    },
}
suffixes = {
    '': 0
}
prefixes = {
    'Dire': {
        'flat_health': 0,
        'flat_defence': 1,
        'flat_attack': 0,
        'scale_health': 1.5,
        'scale_defence': 1,
        'scale_attack': 1.2,
    }
}
abilities = {}

class Unit:

    def __init__(self, creature=None):
        if creature is None:
            creature = {
                'health': 0,
                'defence': 0,
                'attack': 0, }
        self.stats = creature

    def attack(self):
        pass

    def die(self):
        pass


def add_stats(stat, unit):
    unit.health += stat['health']
    unit.attack += stat['attack']
    unit.defence += stat['defence']


def try_add(stat, stat_name, unit):
    if stat_name in stat:
        unit.stats[F'{stat_name}'] += stat[F'{stat_name}']


def try_scale(stat, stat_name, unit):
    if stat_name in stat:
        unit.stats[F'{stat_name}'] *= stat[F'{stat_name}']