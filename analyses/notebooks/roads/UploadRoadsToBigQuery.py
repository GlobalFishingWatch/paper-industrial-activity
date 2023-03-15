# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Upload Roads to Bigquery
#
# This notebook uploads, to BigQuery, a subset of roads from the Global Roads Open Access Data Set (gROADS), v1 (1980 – 2010). Only roads that are within 0.1 degrees (~11km) of the ocean are uploaded to BigQuery.  
#
# The road data can be downloaded from here:
# https://sedac.ciesin.columbia.edu/data/set/groads-global-roads-open-access-v1/data-download
#
# This notebook also requires the dataset Global Oceans and Seas v01 (2021-12-14, 88 MB) from marine regions, here saved as `GOaS_v1_20211214/goas_v01.shp` 

import fiona
from shapely.geometry import shape 
from shapely.geometry import LineString
from shapely.ops import nearest_points
import pandas as pd
import subprocess
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import proplot as pplt

oceans_file = "GOaS_v1_20211214/goas_v01.shp"
ocean = gpd.read_file(oceans_file)
ocean_boundary = ocean.boundary.unary_union

for region in range(1, 8):

    filename = f"GRIP4_Region{region}_vector_shp/GRIP4_region{region}.shp"
    out_filename = f"GRIP4_region{region}"

    shapes = []
    i = 0

    good_shapes = []
    shapes = []
    distance_threshold = 0.1  # 10 km
    chunk_size = 10000

    with fiona.open(filename, "r") as source:
        for a in source:
            i += 1
            if i % chunk_size == 0:
                df = gpd.GeoDataFrame(shapes)
                df["geometry"] = df["geometry"].apply(shape)
                minx, miny, maxx, maxy = df.geometry.total_bounds
                bbox = Polygon(
                    [
                        (minx - adjust, miny - adjust),
                        (minx - adjust, maxy + adjust),
                        (maxx + adjust, maxy + adjust),
                        (maxx + adjust, miny - adjust),
                    ]
                )

                clipped_ocean = ocean_boundary.intersection(bbox)
                if not clipped_ocean.is_empty:
                    nearest_ocean_points = [
                        nearest_points(r, clipped_ocean) for r in df.geometry
                    ]
                    for p, row in zip(nearest_ocean_points, df.iterrows()):
                        distance = p[0].distance(p[1])
                        if distance < distance_threshold:
                            d = row[1]["properties"]
                            d["wkt"] = row[1]["geometry"].wkt
                            d["distance"] = distance
                            good_shapes.append(d)

                shapes = []
                print(i, len(good_shapes))

            shapes.append(a)

    # one more time for the final items...
    df = gpd.GeoDataFrame(shapes)
    df["geometry"] = df["geometry"].apply(shape)
    minx, miny, maxx, maxy = df.geometry.total_bounds
    bbox = Polygon(
        [
            (minx - adjust, miny - adjust),
            (minx - adjust, maxy + adjust),
            (maxx + adjust, maxy + adjust),
            (maxx + adjust, miny - adjust),
        ]
    )

    clipped_ocean = ocean_boundary.intersection(bbox)
    if not clipped_ocean.is_empty:
        nearest_ocean_points = [nearest_points(r, clipped_ocean) for r in df.geometry]
        for p, row in zip(nearest_ocean_points, df.iterrows()):
            distance = p[0].distance(p[1])
            if distance < distance_threshold:
                d = row[1]["properties"]
                d["wkt"] = row[1]["geometry"].wkt
                d["distance"] = distance
                good_shapes.append(d)

    shapes = []
    print(i, len(good_shapes))

    df2 = pd.DataFrame(good_shapes)
    df2.to_csv(f"{out_filename}.csv", index=False)



for region in range(1,8):
    out_filename = f"GRIP4_region{region}"
    df = pd.read_csv(f"{out_filename}.csv")
    d = df#[df.distance<.05]
    print(region, len(d))
    d.to_gbq(f"proj_global_sar.{out_filename}_5km")
