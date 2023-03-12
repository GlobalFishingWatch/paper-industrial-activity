# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
from datetime import datetime, timedelta

# %%
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%
import numpy as np
import pandas as pd
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
import proplot as pplt
import skimage

# %%
# Read gian zip file or pre-proc feather
READZIP = False

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
     between "2017-01-01" and "2021-12-31"
"""

# %%
# NOTE: Read in GIANT csv of all detections,
# created by DownloadAllDetections.ipynb
# this is a huge file... >1.5gb
# df is the master dataframe of all the dections.
# we will filter its size down below.

if READZIP:
    df = pd.read_csv("../data/all_detections.csv.zip")
    df.matched_category.unique()
    df.head()


# %%
# the random numnder is actually a better way to map lots and lots of dots
# basically, for dark vessels, if rand > fishing_score map it as fishing,
# otherwise map it as non-fishing. So, if you have 100 dark detections
# with a score of .501, you will get 50 fishing and 50 non-fishing,
# while just using a threshold of .5 would give you 100 fishing vessels

def get_category_rand(row):
    if (row.matched_category == "matched_nonfishing") or (
        (row.matched_category == "matched_unknown")
        and (row.fishing_score < row.rand)
    ):
        return "matched_nonfishing"

    if (row.matched_category == "unmatched") and (row.fishing_score < 0.5):
        return "dark_nonfishing"

    if (row.matched_category == "matched_fishing") or (
        (row.matched_category == "matched_unknown")
        & (row.fishing_score >= 0.5)
    ):
        return "matched_fishing"

    if (row.matched_category == "unmatched") and (row.fishing_score >= 0.5):
        return "dark_fishing"


if READZIP:
    df["rand"] = np.random.random(len(df))
    df["category_rand"] = df.apply(get_category_rand, axis=1)
    df = df[~df.category_rand.isna()]
    # this drops all detections that have a fishing score that is na and do not match to
    # a vessel. This is a small percent of detections (~1%), but it we can't plot them
    # because we can't assign fishing or non-fishing to them.

# %%
if READZIP:
    df = df.reset_index(drop=True)
    df.to_feather('../data/all_detections_matched_rand.feather')
else:
    df = pd.read_feather('../data/all_detections_matched_rand.feather', use_threads=True)
    # Read previously processed data

df.head()

# %%
# Split into fishing and non-fishing
fishing = ['matched_fishing', 'dark_fishing']                  
nonfish = ['matched_nonfishing', 'dark_nonfishing']
df_fish = df.loc[df.category_rand.isin(fishing)]                  
df_nonf = df.loc[df.category_rand.isin(nonfish)]                  

# %%
f = lambda x: 1 if 'matched' in x else 0
df_fish.loc[:, 'is_matched'] = df_fish.category_rand.map(f)
df_nonf.loc[:, 'is_matched'] = df_nonf.category_rand.map(f)

# %%
df_fish = df_fish.fillna(0)
df_nonf = df_fish.fillna(0)


# %%
def scatter(x, y, c='r', s=1, ax=None, z=10):
    ax.scatter(
        x,
        y,
        c=c,
        s=s,
        zorder=z,
        alpha=alpha,
        edgecolors="none",
        rasterized=True,
        transform=psm.identity,
    )
    return ax


# %%
from shapely import wkt


def get_bbox(min_lon, min_lat, max_lon, max_lat, size=0.05):
    lonc = (min_lon + max_lon) / 2
    latc = (min_lat + max_lat) / 2
    zoom = abs(max_lon - min_lon)
    return (lonc, latc, zoom, size)

bboxes = [
    # eastern north korea eez:
    (126.77,38.07,133.02,42.81, 0.05),
    # northern Japan to russia:
    (136.37,42.5,142.62,46.93, 0.05),
    # near tokyo
    (136.51,33.17,142.76,38.24, 0.05),
    # Shanghai
    (118.82,28.2,125.06,33.55, 0.02),
    # South Japan
    (126.07,27.39,132.32,32.79, 0.05),
    # larger, all of korean peninsula
    (120.91,33.03,133.72,43.25, 0.005),
]

if 0:
    # David bboxes
    B = [get_bbox(*b) for b in bboxes]

else:
    # MPAs centroids
    # df_mpa = pd.read_csv('../data/top_mpas_dark_detections.csv')
    df_mpa = pd.read_csv('../data/no_take_mpas_dark_detections_top15.csv')
    geoms = [wkt.loads(s) for s in df_mpa.mpa_boundary_wkt]
    df_mpa['geometry'] = geoms
    
    geoms = geoms[3:6]

    B = [
        (-90.973796,-0.223786, 5.5, 1.5),  # Galapagos
        (152.614276,-23.021161, 4, 1),  # Great Barrier 
        
        (-71.926763, 9.792800, 5, 2.1),
        (149.190945, -19.874732, .001, 2.1),
        (91.434909, 21.494429, .001, 2.1),
        (142.406820, 43.245659, .001, 2.1),
        (133.202983, 34.205319, .001, 2.1),
        (133.202983, 34.205319, .001, 2.1),  # DUPLICATE
    ]


SAVE = False
matchcolor = "#07A9FD"
darkcolor = "#FD7F0C" 
landcolor = '0.85'
darkfirst = False
alpha = 1

proj_info = [
    [B[0], B[1]],
    [B[2], B[3]],
    [B[4], B[5]],
]

rows, cols, _ = np.shape(proj_info)
rows, cols = 5, 3

plt.rcdefaults()  # <-- important
fig = plt.figure(figsize=(3.75 * cols, 3.15 * rows * 1.1), constrained_layout=True)
fig.patch.set_facecolor('black')
subplot = 0

with psm.context(psm.styles.dark):
    with psm.context({"text.color": "k"}):
        
        for i in range(rows):
            for j in range(cols):

                subplot += 1
                print(f"Subplot: {subplot}")
                
                if 0:
                    # >>>>>>> MPAs >>>>>>>

                    geom = geoms[subplot-1]

                    min_lon, min_lat, max_lon, max_lat = geom.bounds
                    lonc, latc = min_lon/2 + max_lon/2, min_lat/2 + max_lat/2
                    zoom = max_lon - min_lon

                    xc, yc, dx_x_2, size = lonc, latc, zoom, 1/zoom

                    if size < 0.75:
                        size = 0.75 

                    print(
                        df_mpa.mpa_id[subplot-1],
                        df_mpa.mpa_name[subplot-1],
                        df_mpa.country[subplot-1]
                    )
                    print('ZOOM', zoom, size)
                
                    # <<<<<<< MPAs <<<<<<<
                    
                else:
                    
                    xc, yc, dx_x_2, size = proj_info[i][j]
                    
                
                dx = dx_x_2 / 2
                extent = (xc - dx, xc + dx, yc - dx, yc + dx)
                prj = cartopy.crs.LambertAzimuthalEqualArea(xc, yc)
                # prj = psm.find_projection([xc], [yc])
                
                ax = psm.create_map(subplot=(rows, cols, subplot), projection=prj)
                ax.set_extent(extent, crs=psm.identity)
                ax.set_adjustable("datalim")
                
                psm.add_land(ax)
                psm.add_countries(ax)
                psm.add_eezs(ax)
                
                
                if 1:
                    ##### MultiPolygon -> list of polygons

                    for geom in geoms:
                        polys = [list(g.exterior.xy) for g in geom]

                        for poly in polys:
                            ax.plot(*poly, linewidth=0.25, color='w', transform=psm.identity)

                    ##### MultiPolygon -> list of polygons
                
                buf = 6
                df1 = df_fish[
                    (df_fish.lon > extent[0] - buf)
                    & (df_fish.lon < extent[1] + buf)
                    & (df_fish.lat > extent[2] - buf)
                    & (df_fish.lat < extent[3] + buf)
                ]
                
                # To plot on top of each other
                df2 = df1[df1.is_matched == True]
                df3 = df1[df1.is_matched == False]

                z = 1000 if darkfirst else 10
                ax = scatter(df2.lon, df2.lat, c=matchcolor, s=size, ax=ax, z=100)
                ax = scatter(df3.lon, df3.lat, c=darkcolor, s=size, ax=ax, z=z)
                
                vmax = 1500 * 1e3
                base = 0.80
                total = len(df2.lon) + len(df3.lon)
                pdark = round(len(df3.lon) / total, 2) * 100
                frac = round(total / vmax, 2)
                length = frac if frac < base else base
                
                # ===== Colorbar ===== #
                
                if 0:
                    bounds = [0, pdark, 100]
                    cmap = mpl.colors.ListedColormap([darkcolor, matchcolor])
                    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                    smap = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                    cax = ax.inset_axes((0, -0.04, length, 0.0175)) # horizontal
                    # cax = ax.inset_axes((0, -0.04, 0.83, 0.01)) # horizontal
                    # cax = ax.inset_axes((1.03, 0, 0.01, 1))  # vertical
                    
                    cbar = fig.colorbar(
                        smap,
                        cax=cax,
                        ticks=bounds,
                        # ticks=range(0, 110, 10),
                        spacing='proportional',
                        orientation='horizontal',
                    )
                    cbar.outline.set_visible(False)
                    # cbar.ax.set_xticklabels(['0', f"{pdark:.0f}", '100%'])
                    cax.tick_params(
                        length=6,
                        width=2,
                        direction='in',
                        color='w',
                        which='both',
                        labelsize=0,
                        pad=-100
                    )
                    
                    if 0:
                        npts = round(total, -3) * 1e-3
                        xloc = length + 0.01
                        yloc = -0.018
                        space = " " * 11 if npts < 1000 else " " * 13
                        add = ' dark' if subplot != 5 else ''

                        ax.text(
                            xloc,
                            yloc,
                            f"{npts:.0f}k",
                            ha="left",
                            va="top",
                            color='0.3',
                            fontsize=8,
                            transform=ax.transAxes,
                        )
                        ax.text(
                            xloc,
                            yloc,
                            f"{space}{pdark:.0f}%" + add,
                            ha="left",
                            va="top",
                            color=darkcolor,
                            fontsize=8,
                            weight='bold',
                            transform=ax.transAxes,
                        )
                    
                    if 0:  # FIXME
                        ax.text(
                            0,
                            -0.06,
                            "dark fishing",
                            ha="left",
                            va="top",
                            color=darkcolor,
                            fontsize=10,
                            transform=ax.transAxes,
                        )
                        ax.text(
                            length * pdark / 100,
                            -0.06,
                            "publicly tracked",
                            ha="left",
                            va="top",
                            color=matchcolor,
                            fontsize=10,
                            transform=ax.transAxes,
                        )
                    
                if subplot == 2:
                    break
            break

if SAVE:
    plt.savefig("figures/mpas.png", bbox_inches="tight", pad_inches=0.01, dpi=300)

# %%
