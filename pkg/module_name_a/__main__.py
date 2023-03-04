'''
TODO: Edit this
[Example pipeline]

Replace this with a description of what your pipeline does
'''

import argparse
import os

from module_name_a.pipeline import Pipeline
from module_name_a.utils.argparse import pretty_print_args, valid_date

# TODO: Edit these
PIPELINE_VERSION = '0.0.1'
PIPELINE_NAME = '[Example pipeline]'
PIPELINE_DESCRIPTION = (
    'An example pipeline that computes total fishing hours by flag state'
)

# Some optional git parameters provided as environment variables.  Used for logging.
COMMIT_SHA = os.getenv('COMMIT_SHA', '')
COMMIT_BRANCH = os.getenv('COMMIT_BRANCH', '')
COMMIT_REPO = os.getenv('COMMIT_REPO', '')


parser = argparse.ArgumentParser(
    description=f'{PIPELINE_NAME} {PIPELINE_VERSION} - {PIPELINE_DESCRIPTION}'
)
subparsers = parser.add_subparsers(dest='operation', required=True)
fishing_hours_args = subparsers.add_parser(
    'fishing_hours', help='Create the fishing hours table'
)
validate_args = subparsers.add_parser('validate', help='Validate the output table')

parser.add_argument(
    '--test',
    action='store_true',
    help='Test mode - print query and exit',
    default=False,
)


parser.add_argument(
    '--start_date',
    type=valid_date,
    help='Start date. ' 'Format: YYYY-MM-DD (default: %(default)s)',
    default='2016-01-01',
)

parser.add_argument(
    '--end_date',
    type=valid_date,
    help='End date. ' 'Format: YYYY-MM-DD (default: %(default)s)',
    default='2016-12-31',
)

parser.add_argument(
    '--dest_fishing_hours_flag_table',
    type=str,
    help='Destination table for fishing hours by flag state (default: %(default)s)',
    default='world-fishing-827.scratch_public_ttl120.example_fishing_hours_by_flag',
)


fishing_hours_args.add_argument(
    '--source_fishing_effort_table',
    type=str,
    help='Source table for fishing hours (default: %(default)s)',
    default='global-fishing-watch.global_footprint_of_fisheries.fishing_effort',
)

fishing_hours_args.add_argument(
    '--table_description',
    type=str,
    help='Additional text to include in the table description',
    default='Example table from pipe-python-prototype-template.  Ok to delete',
)


args = parser.parse_args()

args.COMMIT_SHA = COMMIT_SHA
args.COMMIT_BRANCH = COMMIT_BRANCH
args.COMMIT_REPO = COMMIT_REPO

print(f'{PIPELINE_NAME} v{PIPELINE_VERSION}')
print(f'{PIPELINE_DESCRIPTION}')
pretty_print_args(args)

args.PIPELINE_VERSION = PIPELINE_VERSION
args.PIPELINE_NAME = PIPELINE_NAME
args.PIPELINE_DESCRIPTION = PIPELINE_DESCRIPTION

pipeline = Pipeline(args)

if __name__ == '__main__':
    result = pipeline.run()

    # exit code=0 indicates success.  Any other value indicates a failure
    exit_code = 0 if result else 1
    exit(exit_code)
