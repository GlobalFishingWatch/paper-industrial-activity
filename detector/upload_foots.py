# -*- coding: utf-8 -*-
"""Download footprints from GCS, convert to WKT and upload to BQ.

Usage (two options):

1. It reads all paramters from the PARAMS_* files generated
    when footprints were exported from GEE to GCS:

    upload_footprints.py path/to/PARAMS_*

2. It takes a DATE and N_DAYS argument (and hardcoded parameters):

    upload_footprints.py 2020-05-01 30

Notes
-----

Unlike detections, the subbucket name in PARAMS for S1A and S1B
is the *same*, regardless of separate processing. So it's fine
to upload from separate dirs/VMs.

To upload footprints only the GCS bucket info is needed.

Links
-----

Original file is located at
https://colab.research.google.com/drive/1iOwqS_fYiaYAbhwaHoUnbvpikCa2h4MK

Referenced to some scripts at
https://github.com/mixuala/colab_utils/blob/master/gcloud.py

"""
import argparse
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from shapely.geometry import shape
from shapely.ops import Polygon, cascaded_union

from core.params import Params
from core.checks import validate_jsons
from core.cloud import download_from_gcs, table_exists, upload_df_to_bq
from core.utils import (
    get_times_from_scene,
    list_files,
    mkdir,
    rmdir,
)

# Remove all downloaded data once uploaded
remove_local_data = True

# Field names for BQ table
# NOTE: Do not change the order!
fields = [
    "scene_id",
    "start_time",
    "end_time",
    "footprint_wkt",
    "footprint_wkt_1km",
    "footprint_wkt_5km",
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Upload footprints to bigquery: GCS->local->BQ"
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


def plot_polygons(polygons):
    """To check we are getting all polygons."""
    for p in polygons:
        plt.plot(*p.exterior.xy)
        if len(p.interiors) > 1:
            for r in p.interiors:
                plt.plot(*r.xy)
    plt.show()


def get_polygons_from_json(jfile, plot=False):
    """GeoJSON -> check indiv. polygons -> list of polygons."""
    polygons = []

    jsn = json.load(open(jfile))

    for item in jsn["features"]:
        geom = shape(item["geometry"])  # dict -> shapely

        if geom.type not in ["Polygon", "MultiPolygon"]:
            continue
        elif geom.type == "Polygon":
            geom = [geom]

        for g in geom:
            if not g.is_empty:
                polygons.append(Polygon(g))

    if plot:
        plot_polygons(polygons)

    return polygons


def geojson_to_wkt(poly):
    """MultiPolygon -> WKT (simplified)>"""
    poly_wkt = poly.wkt
    poly_wkt_1km = poly.simplify(0.01).wkt  # TODO: in m?
    poly_wkt_5km = poly.simplify(0.05).wkt
    return poly_wkt, poly_wkt_1km, poly_wkt_5km


def json_to_dataframe(jfile, fields):
    """Get footprint polygons from GeoJSON -> WKT -> DataFrame.

    NOTE: This df is different from the detection df.

    """
    polygons = get_polygons_from_json(jfile)

    if not polygons:
        return pd.DataFrame()

    footprint = cascaded_union(polygons)

    (
        footprint_wkt,
        footprint_wkt_1km,
        footprint_wkt_5km,
    ) = geojson_to_wkt(footprint)

    scene_id = Path(jfile).stem
    start_time, end_time = get_times_from_scene(scene_id)
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)
    # min_lon, min_lat, max_lon, max_lat = footprint.bounds

    data = [[
        scene_id,
        start_time,
        end_time,
        footprint_wkt,
        footprint_wkt_1km,
        footprint_wkt_5km,
    ]]

    return pd.DataFrame(data, columns=fields)  # 1 row df


def jsons_to_dataframe(jfiles, fields):
    """Get footprint polygons from multiple GeoJSONs -> dataframe."""
    return pd.concat([json_to_dataframe(f, fields) for f in jfiles])


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

    if not table_prefix:
        table_prefix = f"{subbucket}_"
        params.replace(table_prefix=table_prefix)

    date = datetime.strptime(date, "%Y-%m-%d").date()

    YYYYMMDD = f"{date:%Y%m%d}"
    table_name = f"{table_prefix}{YYYYMMDD}"
    table_id = f"{project_id}.{dataset}.{table_name}"
    remote_path = f"{bucket}/{version}/{subbucket}/{YYYYMMDD}"
    local_path = f"{data_dir}/footprints/{YYYYMMDD}"

    check_table(table_id, replace)

    mkdir(local_path)
    download_from_gcs(remote_path, local_path)  # TODO: BytesIO

    jfiles = list_files(local_path, glob="*.geojson")
    jfiles = validate_jsons(jfiles)

    if not jfiles:
        params.replace(is_invalid=True).save()
        return

    # Multiple JSONs -> single DataFrame
    df_foot = jsons_to_dataframe(jfiles, fields)

    upload_df_to_bq(table_id, df_foot, replace)

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
