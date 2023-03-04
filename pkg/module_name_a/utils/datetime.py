from datetime import date, datetime
from typing import Union


def as_date_str(d: Union[date, datetime], format: str = '%Y-%m-%d'):
    '''
    translates dates to strings

    Args:
        d: date or datetime
        format: expected format pattern for date. Defaults to '%Y-%m-%d'.

    Returns:
        date as string
    '''
    result = d.strftime(format)
    return result
