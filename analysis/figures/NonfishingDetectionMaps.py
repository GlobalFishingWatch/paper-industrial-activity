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

# %% [markdown]
# ## Map global fishing activity

# %%
# Load data file generated in FishingDetectionMaps.py 
df = pd.read_feather('../data/all_detections_matched_rand.feather', use_threads=True)
df.head()
print(len(df))

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


def mouse_event(event):
    print(f"{event.xdata:.0f}, {event.ydata:.0f}")


# %%
df_nonf.category_rand.unique()

# %%
# %matplotlib inline

SAVE = True
FONT = 10

scl = 0.933333333

# Sup fig (non-fishing)
b1 = (2.022632,52.222270,  3.25, 0.025 * scl, 'North Sea\nEnglish Channel', 'lr')
b2 = (10.918211,56.029331, 3.5, 0.025 * scl, 'Sweden\nDenmark', 'ur')
b4 = (54.067225,25.957653, 5, 0.075 * scl, 'Persian Gulf', 'ur')
b7 = (122.117374,30.471462, 2.5, 0.04 * scl, 'East China Sea\nShanghai', 'll')
b5 = (103.499679,1.073024, 5.5, 0.05 * scl, 'Strait of Malacca\nMalaysia, Indonesia', 'll')
b10 = (130.249640,34.276966, 3.75, 0.05 * scl, 'South Korea Strait', 'ur')

b3 = (49.721489,28.961714, 3.5, 0.15 * scl)  # Persian Gulf North
b6 = (100.269885,11.373297, 4.5, 0.1 * scl)  # Gulf of Tailand
b8 = (120.458908,35.121065, 1.75, 0.075 * scl)  # East China
b9 = (120.285701,38.740094, 3, 0.05 * scl)  # Bohai Sea

location = {
    'll': (0.02, 0.02, 'left', 'bottom'),
    'lr': (0.98, 0.02, 'right', 'bottom'),
    'ur': (0.98, 0.98, 'right', 'top'),
    'ul': (0.02, 0.98, 'left', 'top'),
}

DF = df_nonf  # NON-FISHING DATA
# matchcolor = "#07A9FD"
# darkcolor = "#FD7F0C" 
darkcolor = "#d7191c"
matchcolor = "#2c7bb6"
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
fig = plt.figure(figsize=(3.75 * cols * scl, 3.15 * rows * 1.1 * scl), constrained_layout=True)
cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
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
                    cax = ax.inset_axes((0, -0.045, length, 0.018)) # horizontal
                    
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
                    
                    ax.text(
                        xloc,
                        yloc,
                        f"{npts:.0f}k",
                        ha="left",
                        va="top",
                        color='0.3',
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
                    
                    if sub == 5:
                        ax.text(
                            0,
                            -0.07,
                            "not tracked",
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
                        
                    if sub == 6:
                        ax.text(
                            0,
                            -0.07,
                            "Non-fishing vessel detections",
                            ha="left",
                            va="top",
                            color='0.1',
                            fontsize=FONT,
                            transform=ax.transAxes,
                        )
                        
                    # ===== labels ===== #
                    
                    # x, y, ha, va = location[loc]
                    # ax.text(x, y, label, ha=ha, va=va, color='0.4', transform=ax.transAxes)
                    
                    # ===== Annotations ===== #
                    
                    if sub == 1:
                        x, y, s = (-133107, 7314, "UNITED\nKINGDOM")
                        ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                        x, y, s = (139399, -143033, "BELGIUM")
                        ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                        x, y, s = (196255, -4432, "NETHERLANDS")
                        ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=90, style='normal', fontsize=FONT)
                    if sub == 2:
                        x, y, s = (-105756, 16783, "DENMARK")
                        ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                        x, y, s = (162437, 139551, "SWEDEN")
                        ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                    if sub == 3:
                        x, y, s = (33612, 202509, "IRAN")
                        ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                        x, y, s = (-38369, -247373, "UNITED ARAB EMIRATES")
                        ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                        x, y, s = (233360, -247373, "OMAN")
                        ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                    if sub == 4:
                        x, y, s = (-104047, 68373, "CHINA")
                        ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                    if sub == 5:
                        x, y, s = (-125136, 215999, "MALAYSIA")
                        ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                        x, y, s = (-188375, -171336, "INDONESIA")
                        ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                    if sub == 6:
                        x, y, s = (-171015, 154616, "SOUTH\nKOREA")
                        ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
                        x, y, s = (73566, -135908, "JAPAN")
                        ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
            
            
if SAVE:
    plt.savefig(
        "./nonfishing_detection_maps_v3.jpg",
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=300
    )

# %%
