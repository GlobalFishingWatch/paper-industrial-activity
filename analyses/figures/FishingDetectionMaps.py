# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
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

# %% [markdown]
# ## Map global fishing activity

# %%
# NOTE: Read in GIANT csv of all detections,
# created by DownloadAllDetections.ipynb
# this is a huge file... >1.5gb
# df is the master dataframe of all the dections.
# we will filter its size down below.
df = pd.read_csv("../data/all_detections_v20230218.csv.gz", compression='gzip')
df = df.rename({'all_detections_v20230218.csv': 'lat'}, axis=1)

# %%
df.head()

# %%
df.matched_category.unique()

# %%
# the random numnder is actually a better way to map lots and lots of dots
# basically, for dark vessels, if rand > fishing_score map it as fishing,
# otherwise map it as non-fishing. So, if you have 100 dark detections
# with a score of .501, you will get 50 fishing and 50 non-fishing,
# while just using a threshold of .5 would give you 100 fishing vessels
df["rand"] = np.random.random(len(df))


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


df["category_rand"] = df.apply(get_category_rand, axis=1)

# %%
# this drops all detections that have a fishing score that is na and do not match to
# a vessel. This is a small percent of detections (~1%), but it we can't plot them
# because we can't assign fishing or non-fishing to them.
df = df[~df.category_rand.isna()]

# %%
df = df.reset_index(drop=True)
df.to_feather('../data/all_detections_matched_rand.feather')
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
# Main fig
b1 = (17.658033,69.736123, 4, 0.05, 'Norwegian Sea', 'ul')
b2 = (4.078690,59.653238, 4.5, 0.075, 'North Sea\nSouthern Norway', 'ur')
b3 = (-6.801825,35.408370, 3.5, 0.05, 'Strait of Gibraltar', 'lr')
b4 = (12.131387, 35.896823, 6, 0.05, 'Strait of Sicily', 'll')
b5 = (122.998371,38.381007, 3.5, 0.015, 'Korea Bay\nNorthern Yellow Sea', 'lr')
b6 = (90.551736,21.064840, 5, 0.04, 'Bay of Bengal\nIndia, Bangladesh, Myanmar', 'll')

# Sup fig
b7 = (-18.535291,65.103852, 4.25, 0.2, 'Iceland', 'lr')
b8 = (-6.390180,48.071477, 7, 0.03, 'Bay of Biscay', 'll')
b9 = (7.422759,54.160562, 2, 0.1, 'North Sea\nNetherlands, Germany', 'lr')
b10 = (17.241459,41.646024, 3.5, 0.075, 'Adriatic Sea', 'ur')
b11 = (128.389955,34.584949, 4.0, 0.05, 'South Korea Strait', 'll')
b13 = (99.953627,8.597927, 4.5, 0.05, 'Gulf of Thailand', 'ur')

b12 = (96.098044,15.941271, 5, 0.05)  # India 
b14 = (53.516775,26.171505, 5, 0.075)  # Persian Gulf

# bX = (-91.346803,28.934702, 6, 0.15)     # G. Mexico 0.15
# bX = (-69.438698,41.878653, 7, 0.09)    # G. Main
# b9 = (120.042745,25.363091, 6, 0.015)  # Taiwan
# b10 = (109.382050,19.168478, 5, 0.03)  # Vietnam
# b11 = (25.086653,39.216986, 3.25, 0.05)  # Greece
# b17 = (14.534464,44.021988, 3.5, 0.06) # N. Adriatic 0.075
# b18 = (21.966780,71.025244, 5, 0.05)  # N. Norway #2

location = {
    'll': (0.02, 0.02, 'left', 'bottom'),
    'lr': (0.98, 0.02, 'right', 'bottom'),
    'ur': (0.98, 0.98, 'right', 'top'),
    'ul': (0.02, 0.98, 'left', 'top'),
}
                    

SAVE = False
matchcolor = "#07A9FD"
darkcolor = "#FD7F0C" 
landcolor = '0.85'
darkfirst = True
alpha = 1

proj_info = [
    # Main fig
    # [b1, b2],
    # [b3, b4],
    # [b5, b6],
    
    # Sup fig
    [b7,   b8],
    [b9,  b10],
    [b11, b13],
    
    # [b13, b14],
    # [b13, b14],
    # [b15, b16],
    # [b17, b18],
]

rows, cols, _ = np.shape(proj_info)

plt.rcdefaults()  # <-- important
fig = plt.figure(figsize=(3.75 * cols, 3.15 * rows * 1.1), constrained_layout=True)
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
                base = 0.83
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
                    
                    # ===== Legend Text ===== #
                    
                    npts = round(total, -3) * 1e-3
                    xloc = length + 0.01
                    yloc = -0.018
                    space = " " * 11 if npts < 1000 else " " * 13
                    add = ' dark' if sub != 5 else ''
                    add2 = ''
                    
                    if sub == 60:
                        add2 = ' detections'
                        space = ' ' * 31
                    
                    ax.text(
                        xloc,
                        yloc,
                        f"{npts:.0f}k" + add2,
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
                    
                    if sub == 5:  # FIXME
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
                        
                        
                    # ===== labels ===== #
                    
                    x, y, ha, va = location[loc]
                    ax.text(x, y, label, ha=ha, va=va, color='0.4', transform=ax.transAxes)
                    
                # if sub < 3:
                #     continue
                # break
            # break

if SAVE:
    plt.savefig("figures/FishingDetectMaps2.png", bbox_inches="tight", pad_inches=0.01, dpi=300)

# %%
