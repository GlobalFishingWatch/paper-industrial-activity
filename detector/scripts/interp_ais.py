"""
Get all the AIS postions that should be close to a scene for a day

Usage:
    interp_ais.py {the_date} {n_days} [version]
    interp_ais.py 2020-01-01 30 v20200820

Note:
    If version is passed and exists, overrides the data on BQ.

"""
import os
import sys
from subprocess import Popen
from .utils import date_range

import pandas as pd

if len(sys.argv) < 3:
    print('Usage: prog.py {the_date} {n_days} [version]')
    sys.exit()

the_date = sys.argv[1]
n_days = sys.argv[2]

if len(sys.argv) > 3:
    version = sys.argv[3]
    process_override = True
else:
    version = "v20200820"
    process_override = False


footprint_vector_table = (
    f"project-id.proj_sentinel1_{version}"
    ".exts_1km"
)

ais_interp_positions_table = (
    f"project-id.proj_sentinel1_{version}"
    ".sentinel_1_ais_interpolated_positions_1km"
)

ais_table = "gfw_research.pipe_v20201001"

image_time_query = (
    "TIMESTAMP_ADD(start_time, INTERVAL "
    "cast(timestamp_diff(end_time, start_time, SECOND)/2 as int64) SECOND)"
)

# Folder with jinja2 files
asset_dir = "assets"


def period_to_colon(table):
    """Convert period between project and dataset to colon."""
    t = table.split(".")
    if len(t) == 2:
        return table
    if len(t) == 3:
        return f"{t[0]}:{t[1]}.{t[2]}"


def data_exists_on_bq(tablename, date):
    """Check if data for a date exists on BQ."""
    q = f"""
    SELECT COUNT(*) number
    FROM `{tablename}`
    WHERE DATE(_partitiontime) = "{date}"
    """
    try:
        df = pd.read_gbq(q, project_id="project-id")
        return df.number.iloc[0] > 0
    except Exception as e:
        print(e)
        return False


def make_table_bq(tablename):
    try:
        tablename = period_to_colon(tablename)
        Popen(
            f"bq mk --time_partitioning_type=DAY {tablename}".split()
        ).wait()
    except Exception as e:
        print(e)
        pass


def interpolate_ais(
    the_date,
    ais_interp_positions_table,
    footprint_vector_table,
    image_time_query,
    asset_dir,
):
    t = the_date.replace("-", "")
    ais_interp_positions_day = ais_interp_positions_table + "\$" + t
    ais_interp_positions_day = period_to_colon(
        ais_interp_positions_day
    )
    command = (
        f"jinja2 {asset_dir}/ais_in_sar_scenes.sql.j2 "
        f"-D thedate='{the_date}' "
        f"-D footprint_vector_table='{footprint_vector_table}' "
        f"-D image_time_query='{image_time_query}' "
        "| "
        "bq query --replace "
        f"--destination_table={ais_interp_positions_day} "
        f"--allow_large_results --use_legacy_sql=false "
    )
    print(command)
    # Popen(command.split()).wait()  # FIXME: Figure out how to PIPE
    os.system(command)


for date in date_range(the_date, n_days):
    if (
        not data_exists_on_bq(ais_interp_positions_table, date)
        or process_override
    ):
        make_table_bq(ais_interp_positions_table)
        interpolate_ais(
            date,
            ais_interp_positions_table,
            footprint_vector_table,
            image_time_query,
            asset_dir,
        )
        print(f'AIS interpolated for {date}')
    else:
        print(f'AIS interpolations exist for {date}')
