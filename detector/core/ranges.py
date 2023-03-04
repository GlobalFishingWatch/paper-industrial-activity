from collections import namedtuple
from datetime import date, datetime, timedelta

import numpy as np
from dateutil.relativedelta import relativedelta


def daterange(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + timedelta(days=n)


def dayrange(date1, n_days):
    for n in range(int(n_days)):
        yield date1 + timedelta(days=n)


def date_range(start, end, return_str=True):
    """Date range generator.

    From (date1, date2) or (date, num_days).
    Returns generator of strings or datetimes.

    start : string or datetime
        e.g. '2020-01-01' or datetime.date(2020, 1, 1)
    end : string, datetime or number
        e.g. '2020-15-30' or 365
    return_str : boolean
        return dates as strings (True) or datetimes (False)

    """
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d")
    if isinstance(end, str) and not end.isnumeric():
        end = datetime.strptime(end, "%Y-%m-%d")

    if isinstance(end, date):
        dr = daterange(start, end)
    else:
        dr = dayrange(start, end)

    if return_str:
        dr = (d.strftime("%Y-%m-%d") for d in dr)
    return dr


def window_range(start, months=12, window=6):
    """Monthly window range generator.

    Return list of montly time intervals: [(t1, t2), (t3, t4)..].
    Start date is the center of the window.
    Returned times are the end points of the window.
    It considers full calendar months (with variable days).

    start : string or datetime
        e.g. '2020-01-01' or datetime.date(2020, 1, 1)
    months : int
        Number of months in range from 'start' date.
    window : int
        Window size in months.

    """
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d")
    # halfw = timedelta(days=int(window * 15))  # fixed length
    halfw = relativedelta(months=window / 2)  # variable length
    dates = [(start + relativedelta(months=k)).date() for k in range(months)]
    # print(f"Window centers {dates}")
    return ((str(d - halfw), str(d + halfw)) for d in dates)


def tile_range(xmin, xmax, ymin, ymax, dx, dy):
    """Tile range generator.

    Corner coordinates (clockwise from ll) for ee.Geometry.

    The BBOX 'width' and 'height' defined by the coords
    must be multiple of tile dx and dy.

    Args:
        xmin, xmax, ymin, ymax (float): region bbox to be tiled.
        dx, dy (float): tile width and height.

    Notes:
        Output must be lists (not ndarrays)!

    """
    nx = int(np.abs(xmax - xmin) / float(dx)) + 1
    ny = int(np.abs(ymax - ymin) / float(dy)) + 1
    x = np.linspace(xmin, xmax, num=nx)
    y = np.linspace(ymin, ymax, num=ny)
    X, Y = np.meshgrid(x, y)

    BBOX = namedtuple("BBOX", "ll ul ur lr")

    for i in range(ny - 1):
        for j in range(nx - 1):
            yield BBOX(
                [X[i, j], Y[i, j]],
                [X[i + 1, j], Y[i + 1, j]],
                [X[i + 1, j + 1], Y[i + 1, j + 1]],
                [X[i, j + 1], Y[i, j + 1]],
            )
