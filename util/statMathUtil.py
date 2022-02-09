import datetime
from time import strftime


def date_to_string(date: datetime):
    return date.strftime('%Y-%m-%d')