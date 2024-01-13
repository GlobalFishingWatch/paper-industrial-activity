# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# DATA  
# - all_detections.csv.zip
# - no_take_mpas_dark_detections_top15.csv
# - galapagos_mpa_boundaries.csv
# - offshore_infra_reclassified_w_region.feather

# %%
import numpy as np
import pandas as pd
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
import proplot as pplt
import skimage

import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from shapely import wkt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta

# %matplotlib inline

# %%
# Read giant zip file or preprocessed feather
READZIP = False

# %% [markdown]
# ## Map global fishing activity

# %%
# NOTE: Read in GIANT csv of all detections,
# created by DownloadAllDetections.ipynb
# this is a huge file... >1.5gb
# df is the master dataframe of all the dections.
# we will filter its size down below.

if READZIP:
    # df = pd.read_feather("../data/all_detections_v20230922.feather")
    df = pd.read_csv("../data/industrial_vessels_v20231013.csv.zip")
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
    # this drops all detections that have a fishing score that is na and do not match to
    # a vessel. This is a small percent of detections (~1%), but it we can't plot them
    # because we can't assign fishing or non-fishing to them.
    df = df[~df.category_rand.isna()]
    df = df.reset_index(drop=True)
    # df.to_feather('../data/all_detections_matched_rand.feather')
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


def mouse_event(event):
    print(f"{event.xdata:.0f}, {event.ydata:.0f}")


# %%
from pathlib import Path
dpath = Path('../data')
list(dpath.iterdir())

# %%
# %matplotlib inline

SAVE = True
FONT = 10

scl = 0.933333333

# MPAs centroids
df_mpa = pd.read_csv('../data/no_take_mpas_dark_detections_top15.csv.zip')
df_mpa = pd.concat([df_mpa, pd.read_csv('../data/galapagos_mpa_boundaries.csv.zip')])
df_infra = pd.read_csv("../data/offshore_infra_reclassified_w_regions_v20230816.csv")

geoms = [wkt.loads(s) for s in df_mpa.mpa_boundary_wkt]
df_mpa['geometry'] = geoms

g_reef = geoms[3:6]
g_galap = geoms[-4:]

B = [
    (-90.973796,-0.223786, 5, 2.5 * scl, g_galap),  # Galapagos
    (152.614276,-23.021161, 4, 1.25 * scl, g_reef),  # Great Barrier 
    
    # (-90.563077,28.869256, 4, 2.5, None),  # Gulf of Mexico
    (-71.451844, 9.935328, 0.75, 1 * scl, None),  # Lake Maracaibo (infra)
    
    # (2.047170,51.852910, 1.25, 1, None),  # English Channel
    (121.401974,33.070388, 1.3, 1.25 * scl, None),  # Shanghai
    
    
    # (7.017328,54.577921, 2.2, 1.25, None),  # Denmark
    # (2.904851,54.853602, 7, 0.5, None),  # North Sea
    # (2.847170,51.582910, 0.35, 3, None),  # Belgian/Nederland farm
    # (1.128954,53.511639, 1.25, 1.75, None),  # UK
    
    # (119.672039,24.596519, 2, 2, None),  # Taiwan
    # (51.921883,26.887918, 5.25, 1, None),  # Persian Gulf
    # (3.684878,56.142141, 10, 0.5, None),  # North Sea
    # (114.055885,21.243702, 5, 0.1, None),  # Macau
    # (-71.571844,9.635328, 1.15, 0.3),  # Lake Maracaibo (vessels)
    # (-91.063077,28.869256, 5.25, 0.25),  # Gulf of Mexico (vessels)
]

letter = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
                    

darkcolor = "#d7191c"
matchcolor = "#2c7bb6"
oilcolor = "#7570b3"
windcolor = "#d95f02"
landcolor = '0.85'
darkfirst = False
alpha = 1

proj_info = [
    [B[0], B[1]],
    [B[2], B[3]],
    # [B[4], B[5]],
]

rows, cols, _ = np.shape(proj_info)

plt.rcdefaults()  # <-- important
fig = plt.figure(figsize=(3.75 * cols * scl, 3.15 * rows * 1.1 * scl), constrained_layout=True)
cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
fig.patch.set_facecolor('white')
subplot = 0

with psm.context(psm.styles.light):
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
                    
                    xc, yc, dx_x_2, size, geoms = proj_info[i][j]
                    
                
                dx = dx_x_2 / 2
                extent = (xc - dx, xc + dx, yc - dx, yc + dx)
                prj = cartopy.crs.LambertAzimuthalEqualArea(xc, yc)
                # prj = psm.find_projection([xc], [yc])
                
                if subplot == 3:
                    prj = 'country.panama'
                
                ax = psm.create_map(subplot=(rows, cols, subplot), projection=prj)
                ax.set_extent(extent, crs=psm.identity)
                ax.set_adjustable("datalim")
                
                psm.add_countries(ax)
                psm.add_eezs(ax)
                psm.add_land(
                    ax=ax,
                    edgecolor='darkgray',
                    facecolor=landcolor,
                    linewidth=0.5,
                )
                
                if geoms:
                    ##### MultiPolygon -> list of polygons
                    for geom in geoms:
                        polys = [list(g.exterior.xy) for g in geom]

                        for poly in polys:
                            ax.plot(*poly, linewidth=0.25, color='0.5', transform=psm.identity)
                
                buf = 6
                if geoms:
                    df1 = df_fish.copy()
                    color1 = matchcolor
                    color2 = darkcolor
                else:
                    df1 = df_infra.copy()
                    color1 = oilcolor
                    color2 = windcolor
                
                df1 = df1[
                    (df1.lon > extent[0] - buf)
                    & (df1.lon < extent[1] + buf)
                    & (df1.lat > extent[2] - buf)
                    & (df1.lat < extent[3] + buf)
                ]
                
                if geoms:
                    # To plot on top of each other
                    df2 = df1[df1.is_matched == True]
                    df3 = df1[df1.is_matched == False]
                else:
                    df2 = df1[df1.label.isin(['oil', 'probable_oil', 'lake_maracaibo'])]
                    df3 = df1[df1.label.isin(['wind', 'probable_wind'])]

                z = 1000 if darkfirst else 10
                ax = scatter(df2.lon, df2.lat, c=color1, s=size, ax=ax, z=100)
                ax = scatter(df3.lon, df3.lat, c=color2, s=size, ax=ax, z=z)
                
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
                        
                if subplot == 1:
                    x, y, s = (39876, -251822, "GALAPAGOS ISLANDS")
                    ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                if subplot == 2:
                    x, y, s = (-151657, -170945, "NORTHEAST\nAUSTRALIA")
                    ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                if subplot == 3:
                    x, y, s = (969048, 1135649, "LAKE\nMARACAIBO")
                    ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                if subplot == 4:
                    x, y, s = (-60395, -57985, "EAST\nCHINA")
                    ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                    
                s = letter[subplot]
                ax.text(.015, .99, s, fontsize=FONT+1, weight='bold', ha='left', va='top', transform=ax.transAxes)
            '''
                if subplot == 1:
                    break
            break
            '''

if SAVE:
    plt.savefig(
        "mpa_infra_maps_v3.jpg",
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=300
    )

# %%
