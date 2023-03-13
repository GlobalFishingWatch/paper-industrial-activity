# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DATA IN THIS NEEDS TO BE UPDATED IF USED

# %%
from datetime import datetime, timedelta
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
import proplot as pplt
import skimage

# %matplotlib inline

# %%
def get_lonlat_from_id(detect_ids):
    lons = [float(d.split(";")[1]) for d in detect_ids]
    lats = [float(d.split(";")[2]) for d in detect_ids]
    return np.array(lons), np.array(lats)


# %% [markdown]
# ## Map global fishing activity

# %%
q = """
select 
  detect_lat,
  detect_lon,
  detect_id,
  length_m,
  score>1e-2 as is_matched_to_ais
from
  proj_global_sar.detections_w_overpasses_v20220805
where
  -- the following is very restrictive on repeated objects
  repeats_100m_180days_forward < 3 and
  repeats_100m_180days_back < 3 and
  repeats_100m_180days_center < 3
  -- get rid of scenes where more than half the detections
  -- are likely noise
  and (scene_detections <=5 or scene_quality > .5)
  and presence > .7
  and extract(date from detect_timestamp) 
     -- between "2017-01-01" and "2021-12-31"
     between "2018-01-01" and "2018-12-31"
"""

# %%
# Load vessel data
# df = pd.read_gbq(q)
# df.to_feather('data/vessel_detections_2018.feather')
df = pd.read_feather("data/vessel_detections_2018.feather", use_threads=True)
# df = pd.read_feather("data/vessel_detections_2021.feather", use_threads=True)
df = df.rename(columns={"detect_lon": "lon", "detect_lat": "lat"})
df.head()

# %%
df["is_matched_to_ais"] = df.is_matched_to_ais.astype(int)


# %%
def hex_to_rgb(value):
    """Converts hex to rgb colours.

    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values
    """
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(
        int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3)
    )


def rgb_to_dec(value):
    """Converts rgb to decimal colours (divides each value by 256).

    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values
    """
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """Create a color map that can be used in heat map.

    If float_list is not provided, colour map graduates
        linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is
        mapped to the respective location in float_list.

    Parameters:
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length
            as hex_list. Must start with 0 and end with 1.

    Returns:
        colour map
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = mpcolors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=256)
    return cmp


palette1 = ["#FF6855", "#E19664", "#C1C674", "#57E69B", "#00FFBB"]  # OrTq1
palette1b = ["#FF705E", "#FF9046", "#CAD33D", "#46FF9E", "#00FFBB"]  # OrTq2
palette2 = ["#FF5DB2", "#FF9063", "#BEC35E", "#3EF35A", "#1EE9B3"]  # PkTq
palette3 = ["#FF705E", "#FF9046", "#CAD33D", "#21FFFF", "#21FFFF"]  # OrBu orig
palette3b = ["#FF705E", "#FF9046", "#CAD33D", "#21FFFF", "#21FFFF"]  # OrBu

mycmap1 = get_continuous_cmap(palette1)
mycmap1b = get_continuous_cmap(palette1b)
mycmap2 = get_continuous_cmap(palette2)
mycmap3 = get_continuous_cmap(palette3)
mycmap3b = get_continuous_cmap(palette3b)

# %%
df = df.fillna(0)

# %%
min_lon, min_lat, max_lon, max_lat = (
    116.7 + 2.5,
    30.6 - 0.5,
    137.1 + 2.5,
    41.6 - 0.5,
)
dx = max_lon - min_lon
dy = max_lat - min_lat
aoi = (min_lon + dx / 2, min_lat + dy / 2, 10.5)

proj_info = [aoi]

fig = plt.figure(figsize=(14, 7.5))

with psm.context(psm.styles.dark):
    with psm.context({"text.color": "white"}):

        xc, yc, dx_x_2 = proj_info[0]
        dx = dx_x_2 / 2
        extent = (xc - dx, xc + dx, yc - dx, yc + dx)
        prj = cartopy.crs.LambertAzimuthalEqualArea(xc, yc)
        ax = psm.create_map(subplot=(1, 1, 1), projection=prj)

        psm.add_land(ax=ax, color="#10151D")
        ax.set_extent(extent, crs=psm.identity)
        ax.set_adjustable("datalim")

        buf = 10
        size = 0.1
        vmin = 0
        vmax = 1

        df1 = df[
            (df.lon > extent[0] - buf)
            & (df.lon < extent[1] + buf)
            & (df.lat > extent[2] - buf)
            & (df.lat < extent[3] + buf)
        ]

        ax.scatter(
            df1.lon,
            df1.lat,
            s=size,
            c=df1.is_matched_to_ais,
            cmap=mycmap3b,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
            rasterized=True,
            transform=psm.identity,
        )

plt.tight_layout()

# plt.savefig("yellow_sea_japan_sar_2018_300dpi.png", bbox_inches="tight", pad_inches=0, dpi=300)

# %%
