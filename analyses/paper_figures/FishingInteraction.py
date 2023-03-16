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
# # DELETE??? NOT NEEDED???

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


# %%
q = """

SELECT *
FROM `world-fishing-827.scratch_pete.trawler_activity_2021_200th_degree` 
where lon_index >= -18867.56
and lat_index >= 5858.24
and lon_index <= -18694.26
and lat_index <= 5973.599999999999

"""

df_oil_bbox = pd.read_gbq(q)

# %%
df_oil_bbox.head()

# %%
from scipy.signal import medfilt2d

lon_min, lat_min, lon_max, lat_max = -94.5, 29.066571, -93.0, 30.060376

oil_raster_total = psm.rasters.df2raster(
    df_oil_bbox,
    "lon_index",
    "lat_index",
    "total_hours",
    xyscale=200,
    per_km2=True,
)


fig = plt.figure(figsize=(12, 10))
norm = mpcolors.LogNorm(vmin=1, vmax=100)

oil_raster_total[oil_raster_total < 10] = 0.0
# oil_raster_total = medfilt2d(oil_raster_total, 3)

with psm.context(psm.styles.dark):

    proj = psm.find_projection(
        df_oil_bbox.lon_index / 200, df_oil_bbox.lat_index / 200
    )
    # extent = (bbox_trawlers[b][0], bbox_trawlers[b][2], bbox_trawlers[b][1], bbox_trawlers[b][3])
    extent = (lon_min, lon_max, lat_min, lat_max)

    with psm.context({"text.color": "white"}):
        ax, im = psm.plot_raster(
            oil_raster_total,
            cmap="presence",
            norm=norm,
            projection=proj.projection,
        )

        ax.set_extent(extent)
        psm.add_countries()
        psm.add_eezs()

        cbax = psm.add_colorbar(
            im, label=r"hours per $\mathregular{km^2}$", width=0.5
        )
        cbax.tick_params(labelsize=16)

        # ax.scatter(infra_points.detect_lon, infra_points.detect_lat, color = 'red', transform = psm.identity, alpha = .5)# zorder=1)

        # plt.savefig(f'{b}_trawler_vessel_activity.png', bbox_inches="tight", dpi = 300)
        plt.show()

# %%

# %%
