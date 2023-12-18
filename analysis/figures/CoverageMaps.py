# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import pandas as pd
import pyseas.maps as psm
import pyseas.contrib as psc
import pyseas.cm
import cmocean

# %matplotlib inline
from datetime import datetime, timedelta
import matplotlib as mpl
import cartopy
import cartopy.crs as ccrs
import cmocean
# -

import sys
sys.path.append('../utils')
from eliminate_ice_string import *
eliminated_locations = eliminate_ice_string()

eliminated_locations = eliminated_locations.replace("detect_lat","lat_index/10")
eliminated_locations = eliminated_locations.replace("detect_lon","lon_index/10")


eliminated_locations = eliminated_locations[len("and not"):]



# +
# q = f"""
# SELECT 
# extract(year from _partitiontime) year,
# extract(year from _partitiontime)*10 + floor(extract(dayofyear from _partitiontime)/366.5*4) quarter,
# lat_index,
# lon_index, 
# {eliminated_locations} in_ice_region,
# sum(overpasses) overpasses 
# FROM `world-fishing-827.proj_sentinel1_v20210924.detect_foot_raster_10` 
# where date(_partitiontime) between "2017-01-01" and "2021-12-31"
# group by lat_index, lon_index, year, quarter, in_ice_region
  
# """
# df = pd.read_gbq(q)
df = pd.read_csv('../data/footprint_rasters_2017_2021.csv.zip')
# -



df.head()

scale = 10

overpass_raster = psm.rasters.df2raster(
        df,
        "lon_index",
        "lat_index",
        "overpasses",
        xyscale=scale,
        per_km2=False,
        origin="lower",
    )

# +

norm = mpcolors.LogNorm(vmin=1, vmax=1000)

with psm.context(psm.styles.light):
    fig = plt.figure(figsize=(12, 8))
    ax, im = psm.plot_raster(
        overpass_raster,
        cmap= cmocean.cm.deep,
        norm=norm,
        origin="lower",
    )
    fig.colorbar(
        im,
        ax=ax,orientation="horizontal",
        fraction=0.02,
        aspect=40,
        pad=0.04,
    label = "Overpasses")
    psm.add_eezs()


#     psm.add_raster(
#         rasters['all_unmatched'],
#         norm=norm,
#         cmap='presence',
#         origin="lower",
#     )
# plt.title("Sentinel-1 Overpasses, 2017-2021",fontsize=15)
plt.savefig(f"./overpasses_2017_2021_light_deep.png",dpi=300, bbox_inches="tight")
# -


