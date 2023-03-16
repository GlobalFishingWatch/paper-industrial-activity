# -*- coding: utf-8 -*-
"""Download, process and upload GeoJSONs (GCS) -> BQ Table.

- Download S1A and/or S1B detection GeoJSONs from GCS
- Update subbucket name if provided as argument
- Update dataset if provided as argument
- Update table_prefix if provided as argument
- Update PARAMS files with new provided params
- Use scene_id for vessels and tile_id for infrastructure
- Generate detect_id from lon/lat/scene_id or tile_id
- Upload to BQ (one table per day), same table for S1A and S1B
- Log successful uploads and empty/invalid params/geojsons

Notes:
    If S1A and S1B were processed separately, the subbucket names
    in PARAMS are different. They need to be uploaded to a common
    subbucket. See Option 2.

    If `region_id` is especified in PARAMS file, it only downloads
    the geojson files that contain `region_id` in their names.
    See Option 3.

Usage:
    FOR VESSELS

    Option 1: All params in a single dir

        upload.py --replace PARAMS_files*

    Option 2: Params in multiple dirs/VMs

        upload.py -t common_table_prefix --append PARAMS_S1A*
        upload.py -t common_table_prefix --append PARAMS_S1B*

    FOR INFRASTRUCTURE

    Option 3: By region_id, single or multiple dirs

        upload.py --append PARAMS_files*

    FOR BOTH

    Option 4: Use original subbucket for GCS but new table for BQ

        upload.py -t new_table_prefix --append PARAMS_files*

    OBS: when a new 'dataset' or 'table_prefix' is passed,
    these fields in PARAMS get updated with the new names.

TODO:
    Use in-memory objects for GeoJSONs (BytesIO).

"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.params import Params
from core.cloud import table_exists, download_from_gcs, upload_df_to_bq
from core.checks import validate_jsons
from core.utils import (
    list_files,
    get_detect_ids,
    get_times_from_scene,
    mkdir,
    rmdir,
)

# Remove all downloaded data once uploaded
remove_local_data = True

# Field names for BQ table
# - Do not change the order!
# - detect_id field is created and added to the table.
# - scene_id gets replaced by tile_id if region_id
#   is not None in params file (for infrastructure).
fields = [
    "detect_lon",
    "detect_lat",
    "scene_id",
    "start_time",
    "end_time",
]


def get_args():
    parser = argparse.ArgumentParser(
        escription="Upload detections to bigquery: GCS->local->BQ"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="PARAMS_*.json",
    )
    parser.add_argument(
        "-d",
        dest="dataset",
        help="BQ dataset name if different from config file",
        default=None,
    )
    parser.add_argument(
        "-t",
        dest="table_prefix",
        help="BQ table prefix if different from subbucket",
        default=None,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-a",
        "--append",
        dest="append",
        action="store_true",
        help="append to BQ table if exists",
        default=None,
    )
    group.add_argument(
        "-r",
        "--replace",
        dest="replace",
        action="store_true",
        help="replace BQ table if exists",
        default=None,
    )
    return parser.parse_args()


def json_to_dataframe(jfile, fields):
    """Get coords/time/id from GeoJSON -> dataframe.

    NOTE: This df is different from the footprint df.

    """
    df = pd.DataFrame()
    jsn = json.load(open(jfile))

    scene_id = Path(jfile).stem
    start_time, end_time = get_times_from_scene(scene_id)
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    # Append rows to empty DataFrame
    for item in jsn["features"]:
        (lon, lat) = item["geometry"]["coordinates"]
        row = [[lon, lat, scene_id, start_time, end_time]]
        df = df.append(pd.DataFrame(row, columns=fields))

    return df  # multi row df


def jsons_to_dataframe(jfiles, fields):
    """Get coords/time/id from multiple GeoJSONs -> dataframe."""
    return pd.concat([json_to_dataframe(f, fields) for f in jfiles])


def add_detect_ids(df):
    col = "tile_id" if "tile_id" in df else "scene_id"
    ids = df[col].values
    xs = df.detect_lon.values
    ys = df.detect_lat.values
    did = get_detect_ids(xs, ys, ids)
    df["detect_id"] = did
    return df


def check_table(table_id, replace):
    if replace is None and table_exists(table_id):
        msg = f"{table_id} exists, use --replace or --append"
        raise ValueError(msg)
    else:
        print(f"{table_id} will be created if needed")


def main(pfile, replace=None, **kwargs):

    params = Params(pfile).replace(**kwargs)

    if params.is_uploaded or params.is_invalid:
        print(f"Skipping {pfile}, is uploaded or invalid")
        return

    date = params.date
    project_id = params.project_id
    bucket = params.bucket
    version = params.version
    subbucket = params.subbucket
    dataset = params.dataset
    table_prefix = params.table_prefix
    data_dir = params.data_dir
    region_id = params.region_id

    if not table_prefix:
        table_prefix = f"{subbucket}_"
        params.replace(table_prefix=table_prefix)

    date = datetime.strptime(date, "%Y-%m-%d").date()

    YYYYMMDD = f"{date:%Y%m%d}"
    table_name = f"{table_prefix}{YYYYMMDD}"
    table_id = f"{project_id}.{dataset}.{table_name}"
    remote_path = f"{bucket}/{version}/{subbucket}/{YYYYMMDD}"
    local_path = f"{data_dir}/detections/{YYYYMMDD}"

    if region_id:
        remote_path += f"/*{region_id}*"
        local_path += f"/{region_id}/"
        # NOTE: Download by 'region_id' instead of by 'date'
        # Only get files with 'region_id' in the name from
        # the SAME 'YYYYMMDD' folder on GCS. This way we only
        # download/upload the files generated by the VM in
        # question (in case of multiple VMs running).

        if "scene_id" in fields:
            fields[fields.index("scene_id")] = "tile_id"

    check_table(table_id, replace)

    mkdir(local_path)
    download_from_gcs(remote_path, local_path)  # TODO: BytesIO

    jfiles = list_files(local_path, glob="*.geojson")
    jfiles = validate_jsons(jfiles)

    if not jfiles:
        params.replace(is_invalid=True).save()
        return

    # Multiple JSONs -> single DataFrame
    df_detect = jsons_to_dataframe(jfiles, fields)
    df_detect = add_detect_ids(df_detect)

    upload_df_to_bq(table_id, df_detect, replace)

    params.replace(is_uploaded=True).save()

    if remove_local_data:
        rmdir(local_path)


if __name__ == "__main__":

    args = get_args()
    files = args.files
    append = args.append
    replace = args.replace

    kwargs = {
        "dataset": args.dataset,
        "table_prefix": args.table_prefix,
    }

    if append:
        replace = not append

    [main(f, replace, **kwargs) for f in files]
