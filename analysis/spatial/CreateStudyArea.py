# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Get polgyons that have at least 30 overpasses from 2017 to 2021
#
# This produces a raster of all locations, at 100th of a degree, that have at least 30 overpasses across our study period and turns them into a series of polygons, and then saves those polygons in a file. 
#
#

import numpy as np
import pandas as pd
import geopandas as gpd
import affine
import rasterio as rio
import rasterio.features
import pyseas.maps as psm
import shapely
from shapely.geometry import shape


# ## Download csv of data at 100th of a degree
#
# This takes a long time
#

# use the standard for eliminating ice locations.
import sys
sys.path.append('../utils')
from eliminate_ice_string import *
from proj_id import project_id
eliminated_locations = eliminate_ice_string()


# +
# this is... high res
# takes a long time
scale = 100

eliminated_locations_index = eliminated_locations.replace(
    "detect_lon", "(lon_index/200 + 1/400)"
).replace("detect_lat", "(lat_index/200 + 1/400)")

q = f"""with overpasses_over_30 as 

(select 
lat_index,
lon_index,
sum(overpasses) overpasses
from `{project_id}.proj_global_sar.overpasses_200_by_year_v20221031`  
where year between 2017 and 2021
{eliminated_locations_index}
group by lat_index,lon_index
having overpasses >= 30)


select 
distinct
floor(lat_index/200*{scale}) lat_index,
floor(lon_index/200*{scale}) lon_index,
1 as value
from overpasses_over_30

"""

df_study = pd.read_gbq(q)

# +
raster = psm.rasters.df2raster(
        df_study,
        "lon_index",
        "lat_index",
        "value",
        xyscale=scale,
        per_km2=False,
        origin="upper",
    )

cellsize = 1/scale
max_lat = 90
min_lat = -90
min_lon = -180
max_lon = 180
num_lats = (max_lat - min_lat) * scale
num_lons = (max_lon - min_lon) * scale

transform = affine.Affine(float(cellsize), 0, -180,
                               0, -float(cellsize), max_lat)

shapes = []
for shp, val in rasterio.features.shapes(np.float32(raster), transform=transform):
    shapes.append(shape(shp))
# -


df_shapes = gpd.GeoDataFrame({'study_area':shapes})
df_shapes['study_area_02'] = df_shapes.study_area.apply(lambda x: x.simplify(.02))
df_shapes['study_area_05'] = df_shapes.study_area.apply(lambda x: x.simplify(.05))

df_shapes.to_csv("../data/study_area.csv",index=False)


