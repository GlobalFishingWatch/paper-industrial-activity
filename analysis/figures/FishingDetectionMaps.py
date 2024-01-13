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
from pathlib import Path

# %% [markdown]
# ## Map global fishing activity

# %%
# df = pd.read_feather("../data/all_detections_v20231013.feather")
df = pd.read_csv("../data/industrial_vessels_v20231013.csv.zip")


# %%
df.head()
print(len(df))

# %%
df.matched_category.unique()

# %%
# the random numnder is actually a better way to map lots and lots of dots
# basically, for dark vessels, if rand > fishing_score map it as fishing,
# otherwise map it as non-fishing. So, if you have 100 dark detections
# with a score of .501, you will get 50 fishing and 50 non-fishing,
# while just using a threshold of .5 would give you 100 fishing vessels

data_rand = '../data/all_detections_matched_rand.feather'


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

    
# Check if file exists, if not create it
if not Path(data_rand).is_file():
    df["rand"] = np.random.random(len(df))
    df["category_rand"] = df.apply(get_category_rand, axis=1)

    # this drops all detections that have a fishing score that is na and do not match to
    # a vessel. This is a small percent of detections (~1%), but it we can't plot them
    # because we can't assign fishing or non-fishing to them.
    df = df[~df.category_rand.isna()]

    df = df.reset_index(drop=True)
    df.to_feather(data_rand)
    
else:
    df = pd.read_feather(data_rand, use_threads=True)

df.head()

# %%

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
import os
import geopandas as gpd
import warnings
from pyseas import props
from pyseas.maps import identity

root = psm.core.root
_eezs = {}

def add_eezs(
    ax=None,
    *,
    facecolor="none",
    edgecolor=None,
    linewidth=None,
    alpha=1,
    include=None,
    exclude=None
):
    """Add EEZs to an existing map

    Parameters
    ----------
    ax : matplotlib axes object, optional
    facecolor : str, optional
    edgecolor: str or tuple, optional
        Can be styled with 'pyseas.eez.bordercolor'
    linewidth: float, optional
        Can be styled with 'pyseas.eez.linewidth'
    alpha: float, optional
    included: optional, set-like: if set, filter lines to only those that are in included
    excluded: optional, set-like: if set, remove lines that are in excluded


    Returns
    -------
    FeatureArtist
    """
    if ax is None:
        ax = plt.gca()
    path = os.path.join(root, "pyseas/data/eezs/eez_boundaries_v11.gpkg")
    if path not in _eezs:
        try:
            with warnings.catch_warnings():
                # Suppress useless RuntimeWarning from geopandas when reading EEZs
                warnings.simplefilter("ignore")
                _eezs[path] = gpd.read_file(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Eezs must be installed into the `pyseas/data/` directory"
            )

    eezs = _eezs[path]
    if include is not None:
        eezs = eezs[eezs.LINE_TYPE.isin(include)]
    if exclude is not None:
        eezs = eezs[~eezs.LINE_TYPE.isin(exclude)]
    edgecolor = edgecolor or plt.rcParams.get(
        "pyseas.eez.bordercolor", props.dark.eez.color
    )
    linewidth = linewidth or plt.rcParams.get("pyseas.eez.linewidth", 0.4)

    return ax.add_geometries(
        eezs.geometry,
        crs=identity,
        alpha=alpha,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )


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

def mouse_event(event):
    print(f"{event.xdata:.0f}, {event.ydata:.0f}")

# %%

# %matplotlib inline

SAVE = True
MAIN = False
FONT = 10




scl = 0.933333333


if MAIN:
    # Main fig
    b1 = (17.658033,69.736123, 4, 0.05 * scl, 'Northern Norway', 'lr')
    b2 = (4.078690,59.653238, 4.5, 0.075 * scl, 'Southern Norway', 'ur')
    b3 = (-6.801825,35.408370, 3.5, 0.05 * scl, 'Strait of Gibraltar', 'lr')
    b4 = (12.131387, 35.896823, 6, 0.05 * scl, 'Strait of Sicily', 'll')
    b5 = (122.998371,38.381007, 3.5, 0.015 * scl, 'Korea Bay', 'lr')
    b6 = (90.551736,21.064840, 5, 0.04 * scl, 'Bay of Bengal', 'll')
    
    subplots = [
        [b1, b2],
        [b3, b4],
        [b5, b6],
    ]
        
else:
    # Sup fig
    b7 = (-18.535291,65.103852, 4.25, 0.2 * scl, 'Iceland', 'lr')
    b8 = (-6.390180,48.071477, 7, 0.03 * scl, 'Bay of Biscay', 'll')
    b9 = (7.422759,54.160562, 2, 0.1 * scl, 'North Sea\nNetherlands, Germany', 'lr')
    b10 = (17.241459,41.646024, 3.5, 0.075 * scl, 'Adriatic Sea', 'ur')
    b11 = (128.389955,34.584949, 4.0, 0.05 * scl, 'South Korea Strait', 'll')
    b13 = (99.953627,8.597927, 4.5, 0.05 * scl, 'Gulf of Thailand', 'ur')
    
    subplots = [
        [b7,   b8],
        [b9,  b10],
        [b11, b13],
    ]

location = {
    'll': (0.02, 0.02, 'left', 'bottom'),
    'lr': (0.98, 0.02, 'right', 'bottom'),
    'ur': (0.98, 0.98, 'right', 'top'),
    'ul': (0.02, 0.98, 'left', 'top'),
}

letter = {
    1: 'a',
    2: 'b',
    3: 'c',
    4: 'd',
    5: 'e',
    6: 'f',
}
                    
# matchcolor = "#07A9FD"
# darkcolor = "#FD7F0C" 
darkcolor = "#d7191c"
matchcolor = "#2c7bb6"
landcolor = '0.85'
darkfirst = True
alpha = 1

proj_info = subplots
rows, cols, _ = np.shape(proj_info)

plt.rcdefaults()  # <-- important
fig = plt.figure(figsize=(3.75 * cols * scl, 3.15 * rows * 1.1 * scl), constrained_layout=True)
cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
fig.patch.set_facecolor('white')
sub = 0

with psm.context(psm.styles.light):
    with psm.context({"text.color": "k"}):
        
        for i in range(rows):
            for j in range(cols):

                sub += 1
                print(f"sub: {sub}")

                xc, yc, dx_x_2, size, label, loc = proj_info[i][j]
                dx = dx_x_2 / 2
                extent = (xc - dx, xc + dx, yc - dx, yc + dx)
                prj = cartopy.crs.LambertAzimuthalEqualArea(xc, yc)
                
                if sub == 3:  # FIXME
                    prj = "regional.european_union"
                    
                if sub == 8:  # FIXME
                    prj = psm.find_projection([xc], [yc])
                
                ax = psm.create_map(subplot=(rows, cols, sub), projection=prj)
                ax.set_extent(extent, crs=psm.identity)
                ax.set_adjustable("datalim")
                
                psm.add_land(
                    ax=ax,
                    edgecolor='darkgray',
                    facecolor=landcolor,
                    linewidth=0.5,
                )
                # psm.add_countries(ax, linewidth=1)
                # psm.add_eezs(ax, linewidth=1)
                psm.add_countries(ax, linewidth=0.75, facecolor=(0, 0, 0, 0))
                add_eezs(ax, linewidth=0.75, exclude={"Straight Baseline"})
                
                # FIXME: don't use buffer for percentages???
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
 
                # z = 1000 if darkfirst else 10
                # ax = scatter(df2.lon, df2.lat, c=matchcolor, s=size, ax=ax, z=100)
                # ax = scatter(df3.lon, df3.lat, c=darkcolor, s=size, ax=ax, z=z)
            
                df1 = df1.sample(frac=1)
            
                color = [matchcolor if m else darkcolor for m in df1.is_matched]
                ax = scatter(df1.lon, df1.lat, c=color, s=size, ax=ax, z=1000)
                
                vmax = 1500 * 1e3
                base = 0.76
                total = len(df2.lon) + len(df3.lon)
                pdark = round(len(df3.lon) / total, 2) * 100
                frac = round(total / vmax, 2)
                length = frac if frac < base else base
                
                # ===== Colorbar ===== #
                
                if 1:
                    bounds = [0, pdark, 100]
                    cmap = mpl.colors.ListedColormap([darkcolor, matchcolor])
                    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                    smap = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                    cax = ax.inset_axes((0, -0.045, length, 0.018)) # horizontal
                    
                    cbar = fig.colorbar(
                        smap,
                        cax=cax,
                        ticks=bounds,
                        spacing='proportional',
                        orientation='horizontal',
                    )
                    cbar.outline.set_visible(False)
                    cax.tick_params(
                        length=6,
                        width=2,
                        direction='in',
                        color='w',
                        which='both',
                        labelsize=0,
                        pad=-100
                    )
                    
                    # ===== Legend Text ===== #
                    
                    npts = round(total, -3) * 1e-3
                    xloc = length + 0.01
                    yloc = -0.018
                    space = " " * 11 if npts < 1000 else " " * 13
                    
                    ax.text(
                        xloc,
                        yloc,
                        f"{npts:.0f}k",
                        ha="left",
                        va="top",
                        color='0.1',
                        fontsize=FONT,
                        transform=ax.transAxes,
                    )
                    ax.text(
                        xloc,
                        yloc,
                        f"{space}{pdark:.0f}%",
                        ha="left",
                        va="top",
                        color=darkcolor,
                        fontsize=FONT,
                        weight='normal',
                        transform=ax.transAxes,
                    )
                    
                    if sub == 5:  # FIXME
                        ax.text(
                            0,
                            -0.07,
                            "not publicly tracked",
                            ha="left",
                            va="top",
                            color=darkcolor,
                            fontsize=FONT,
                            transform=ax.transAxes,
                        )
                        ax.text(
                            length * pdark / 100,
                            -0.07,
                            "publicly tracked",
                            ha="left",
                            va="top",
                            color=matchcolor,
                            fontsize=FONT,
                            transform=ax.transAxes,
                        )
                        
                    if sub == 6:  # FIXME
                        ax.text(
                            0,
                            -0.07,
                            "Fishing vessel detections",
                            ha="left",
                            va="top",
                            color='0.1',
                            fontsize=FONT,
                            transform=ax.transAxes,
                        )
                        
                    # ===== labels ===== #
                    
                    # x, y, ha, va = location[loc]
                    # ax.text(x, y, label, ha=ha, va=va, color='0.4', transform=ax.transAxes)
                    
                    # ===== annotations ===== #
                    
                    if MAIN:
                        if sub == 1:
                            x, y, s = (184379, -21463, "NORWAY")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (112586, -189458, "SWEDEN")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (-174483, 195000, "Norwegian Sea")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='italic', fontsize=FONT)
                            x, y, s = (49820, 195000, "Barents Sea")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='italic', fontsize=FONT)
                            x, y, s = (-140669, -34210, "Shelf break")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=32, style='normal', fontsize=FONT)
                        if sub == 2:
                            x, y, s = (207092, 36308, "NORWAY")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (-80400, -189447, "North Sea")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='italic', fontsize=FONT)
                            x, y, s = (11571, 11937, "Norwegian channel")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=-84, style='normal', fontsize=FONT)
                        if sub == 3:
                            x, y, s = (-1738323, -1294282, "SPAIN")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (-1764817, -1587187, "MOROCCO")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (-2030256, -1300127, "Shelf break")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=-15, style='normal', fontsize=FONT)
                            x, y, s = (-2049222, -1561267, "Shelf break")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=13, style='normal', fontsize=FONT)
                        if sub == 4:
                            x, y, s = (176086, 186752, "SICILY")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (-228869, -61792, "TUNISIA")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (255668, -195033, "Mediterranean Sea")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='italic', fontsize=FONT)
                            x, y, s = (-222357, 278567, "Ocean banks")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=22, style='normal', fontsize=FONT)
                            x, y, s = (188080, -24879, "Ocean banks")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=-36, style='normal', fontsize=FONT)
                        if sub == 5:
                            x, y, s = (180608, 174358, "NORTH\nKOREA")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (-70773, 151847, "CHINA")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (7741, 5000, "Aquaculture\nfarms")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (66522, -101504, "Yellow Sea")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='italic', fontsize=FONT)
                        if sub == 6:
                            x, y, s = (-259893, 202107, "INDIA")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (-88790, 205671, "BANGLADESH")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (255197, -8207, "MYANMAR")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (-68838, -194283, "Bay of Bengal")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='italic', fontsize=FONT)
                            x, y, s = (-221985, -55337, "Canyon system")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=17, style='normal', fontsize=FONT)
                            x, y, s = (28552, -50000, "Shelf break")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=-9, style='normal', fontsize=FONT)
                    else:
                        if sub == 1:
                            x, y, s = (12435, -28042, "ICELAND")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                        if sub == 2:
                            x, y, s = (332957, 1452, "FRANCE")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (308416, 350730, "GB")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                        if sub == 3:
                            x, y, s = (-592200, 378223, "NETHERLANDS")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (-474504, 378223, "GERMANY")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                        if sub == 4:
                            x, y, s = (-120181, -91751, "ITALY")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (208246, -70840, "ALBANIA")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=90, style='normal', fontsize=FONT)
                            x, y, s = (159044, 130891, "MONTENEGRO")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                        if sub == 5:
                            x, y, s = (-42564, 144128, "SOUTH\nKOREA")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            x, y, s = (202636, -135486, "JAPAN")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                        if sub == 6:
                            x, y, s = (-95627, 16850, "THAILAND")
                            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                            
                            
                    # s = letter[sub]
                    # ax.text(.015, .99, s, fontsize=FONT+1, weight='bold', ha='left', va='top', transform=ax.transAxes)
                        
                        
if SAVE:
    if MAIN:
        name = "fishing_detection_maps_v3.jpg"
    else:
        name = "fishing_detection_maps_sup_v3.jpg"
    plt.savefig(name, bbox_inches="tight", pad_inches=0.01, dpi=300)

# %%
