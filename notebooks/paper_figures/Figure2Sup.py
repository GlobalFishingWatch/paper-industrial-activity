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

# %% [markdown]
# ## Map global fishing activity

# %%
# Load data file generated in Figure2v5.py
df = pd.read_feather('../data/all_detections_matched_rand.feather', use_threads=True)
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
def hex_to_rgb(value):
    """Converts hex to rgb colours.

    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values
    """
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """Converts rgb to decimal colours (divides each value by 256).

    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values
    """
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None, n=256):
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
    cmp = mpcolors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=n)
    return cmp


palette3 = ['#FF6855', '#E19664', '#C1C674', '#57E69B', '#00FFBB']  # OrTq1
# palette2 = ['#FF705E', '#FF9046', '#CAD33D', '#46FF9E', '#00FFBB']  # OrTq2
# palette = ['#FF5DB2', '#FF9063', '#BEC35E', '#3EF35A', '#1EE9B3']  # PkTq
# palette3 = ['#FF5DB2', '#FF9063', '#3EF35A', '#1EE9B3']  # PkTq mod
# palette2 = ["#ff6854", "#FF705E", "#46FF9E", "#00FFBB"]  # OrTq2 mod
# palette2 = ['#FF705E', '#FF9046', '#CAD33D', '#21FFFF', '#21FFFF']  # OrBu
# palette3 = ['#FFFF0D', '#21FFFF']
# palette3 = ['#21FFFF', '#FF5DB2']


palette0 = ["#4493EC", "#EA6E42"] # Blu-Orange
palette2 = ["#ca0020", "#f4a582","#f7f7f7","#92c5de","#0571b0",]  # Red-Blue
# palette = ["#7b3294","#c2a5cf","#f7f7f7","#a6dba0","#008837",] # Green-Purple

palette = [
# '#bae4bc',
'#7bccc4',
'#43a2ca',
'#0868ac',
'#000000',
]

palette1 = [
'#b2182b',
# '#d6604d',
# '#f4a582',
# '#fddbc7',
# '#f7f7f7',
# '#d1e5f0',
# '#92c5de',
# '#4393c3',
'#2166ac',
]

# palette1 = ['#1D8F64', '#DE0077']
# palette2 = ['#40A33A', '#843692']
palette4 = ['#921637', '#FFFFD9', '#12275E']

mycmap0 = get_continuous_cmap(palette0, n=2)
mycmap1 = get_continuous_cmap(palette1, n=2)
mycmap2 = get_continuous_cmap(palette2)
mycmap4 = get_continuous_cmap(palette4)

# %%
df_fish = df_fish.fillna(0)
df_nonf = df_nonf.fillna(0)

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
df_nonf.category_rand.unique()

# %%
# Sup fig (non-fishing)
b1 = (2.022632,52.222270,  3.25, 0.025, 'North Sea\nEnglish Channel', 'lr')
b2 = (10.918211,56.029331, 3.5, 0.025, 'Sweden\nDenmark', 'ur')
b4 = (54.067225,25.957653, 5, 0.075, 'Persian Gulf', 'ur')
b7 = (122.117374,30.471462, 2.5, 0.04, 'East China Sea\nShanghai', 'll')
b5 = (103.499679,1.073024, 5.5, 0.05, 'Strait of Malacca\nMalaysia, Indonesia', 'll')
b10 = (130.249640,34.276966, 3.75, 0.05, 'South Korea Strait', 'ur')

b3 = (49.721489,28.961714, 3.5, 0.15)  # Persian Gulf North
b6 = (100.269885,11.373297, 4.5, 0.1)  # Gulf of Tailand
b8 = (120.458908,35.121065, 1.75, 0.075)  # East China
b9 = (120.285701,38.740094, 3, 0.05)  # Bohai Sea

location = {
    'll': (0.02, 0.02, 'left', 'bottom'),
    'lr': (0.98, 0.02, 'right', 'bottom'),
    'ur': (0.98, 0.98, 'right', 'top'),
    'ul': (0.02, 0.98, 'left', 'top'),
}

DF = df_nonf  # NON-FISHING DATA
SAVE = True
matchcolor = "#07A9FD"
darkcolor = "#FD7F0C" 
landcolor = '0.85'
darkfirst = True
alpha = 1

proj_info = [
    [b1, b2],
    [b4, b7],
    [b5, b10],
    
    # [b7, b8],
    # [b9, b10],
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
                print(f"Sub: {sub}")

                xc, yc, dx_x_2, size, label, loc = proj_info[i][j]
                dx = dx_x_2 / 2
                extent = (xc - dx, xc + dx, yc - dx, yc + dx)
                prj = cartopy.crs.LambertAzimuthalEqualArea(xc, yc)
                
                if sub == 30:  # FIXME
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
                df1 = DF[
                    (DF.lon > extent[0] - buf)
                    & (DF.lon < extent[1] + buf)
                    & (DF.lat > extent[2] - buf)
                    & (DF.lat < extent[3] + buf)
                ]
                
                # To plot on top of each other
                df2 = df1[df1.is_matched == True]
                df3 = df1[df1.is_matched == False]

                z = 1000 if darkfirst else 10
                ax = scatter(df2.lon, df2.lat, c=matchcolor, s=size, ax=ax, z=100)
                ax = scatter(df3.lon, df3.lat, c=darkcolor, s=size, ax=ax, z=z)
                
                vmax = 1100 * 1e3
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
                    
                    npts = round(total, -3) * 1e-3
                    xloc = length + 0.01
                    yloc = -0.018
                    space = " " * 11 if npts < 1000 else " " * 13
                    add = ' dark' if sub != 5 else ''
                    
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
                    
                    if sub == 5:
                        ax.text(
                            0,
                            -0.06,
                            "dark activity",
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
    plt.savefig("figures/fig2v2sup2.png", bbox_inches="tight", pad_inches=0.01, dpi=300)

# %%
