"""
Rasterize footprint: Create Overpass Raster for N days at ~500m

Usage:
    rasterize.py {the_date} {n_days} [version]
    rasterize.py 2020-01-01 30 v20200820

Note:
    If version is passed and exists, overrides the data on BQ.

"""
import os
import sys
from subprocess import Popen
from utils import date_range

import pandas as pd

if len(sys.argv) < 3:
    print('Usage: prog.py {the_date} {n_days} [version]')
    sys.exit()

the_date = sys.argv[1]
n_days = int(sys.argv[2])

if len(sys.argv) > 3:
    version = sys.argv[3]
    process_override = True
else:
    version = "v20200820"
    process_override = False

# table that has the outlines of the scenes come from
footprint_vector_table = (
    f"project-id.proj_sentinel1_{version}"
    ".exts_1km"
)

# table where we will put the raster outputs
footprint_raster_table = (
    f"project-id.proj_sentinel1_{version}"
    ".sentinel_1_footprints_500m_1km"
)

# the resolution, in degrees^-1.
# so 200 means each grid cell is 1/200th of a degree which is ~500m
# this script can do higher and lower resolutions!
one_over_cellsize = 200

# Folder with jinja2 files
asset_dir = "assets"


def period_to_colon(tablename):
    """Convert period between project and dataset to colon."""
    t = tablename.split(".")
    if len(t) == 2:
        return tablename
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


def rasterize_footprint(
    the_date,
    footprint_vector_table,
    footprint_raster_table,
    one_over_cellsize,
    asset_dir,
):
    t = the_date.replace("-", "")
    footrpint_vector_table_day = footprint_vector_table + t
    footprint_raster_table_day = footprint_raster_table + "\$" + t
    footprint_raster_table_day = period_to_colon(
        footprint_raster_table_day
    )
    command = (
        f"jinja2 {asset_dir}/raster.sql.j2 "
        f"-D one_over_cellsize='{one_over_cellsize}' "
        f"-D footrpint_vector_table_day='{footrpint_vector_table_day}' "
        "| "
        "bq query --replace "
        f"--destination_table={footprint_raster_table_day} "
        f"--allow_large_results --use_legacy_sql=false "
    )
    print(command)
    # Popen(command.split()).wait()  # FIXME: Figure out how to PIPE
    os.system(command)


for date in date_range(the_date, n_days):
    print(date)
    if (
        not data_exists_on_bq(footprint_raster_table, date)
        or process_override
    ):
        make_table_bq(footprint_raster_table)
        rasterize_footprint(
            date,
            footprint_vector_table,
            footprint_raster_table,
            one_over_cellsize,
            asset_dir,
        )
        print(f'Rasterization done for {date}')
    else:
        print(f'Rasters already exist for {date}')
