'''
Utilities for use with argparse
'''

import argparse
from datetime import datetime


def valid_date(s: str) -> datetime:
    '''
    Use with argparse to validate a date parameter

    Example Usage:
    parser.add_argument('--start_date', type=valid_date,
                    help='Start date for the pipeline. '
                         'Format: YYYY-MM-DD (default: %(default)s)',
                    default='2019-01-01')

    Args:
        s: string of date

    Raises:
        ArgumentTypeError: not a valid date

    Returns:
        datetime object
    '''
    try:
        return datetime.strptime(s, '%Y-%m-%d').date()
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def pretty_print_args(args):
    print('Executing with parameters:')
    print('\n'.join(f'  {k}={v}' for k, v in vars(args).items()))
