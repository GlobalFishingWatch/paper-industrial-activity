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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['figure.dpi'] = 600

# %%
import numpy as np
import pandas as pd
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
import proplot as pplt
import skimage
import shapely
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta

from cmaps import *

# %matplotlib inline

# %%
# use the standard for eliminating ice locations.
import sys
sys.path.append('../utils')
from eliminate_ice_string import eliminate_ice_string

eliminated_locations = eliminate_ice_string()


# %%
FONT = 8

palette7 = [
    ## Darker
    # "#ca0020",
    # "#f4a582",
    # "#92c5de",
    # "#0571b0",
    ## Brighter
    "#d7191c",
    "#fdae61",
    "#abd9e9",
    "#2c7bb6",
]

# red = "#ca0020"
# blue = "#0571b0"
red = "#d7191c"
blue = "#2c7bb6"

mycmap7 = get_continuous_cmap(palette7, n=4)


# %% [markdown]
# ## Map global fishing activity

# %%
def map_bivariate(
    grid_total,
    grid_ratio,
    title,
    vmax=0.2,
    a_vmin=0.1,
    a_vmax=10,
    eez_linewidth=0.4,
    eez_color="#5c5c5c",
    cmap=None,
    cbar=False,
    psm=None,
    ax=None,
):

    # cmap = psm.cm.bivariate.orange_blur.reversed()
    cmap_bi = psm.cm.bivariate.TransparencyBivariateColormap(cmap)
    norm1 = mpcolors.Normalize(vmin=0.0, vmax=vmax, clip=True)
    norm2 = mpcolors.LogNorm(vmin=a_vmin, vmax=a_vmax, clip=True)

    img = psm.add_bivariate_raster(
        grid_ratio, grid_total, cmap_bi, norm1, norm2, origin="lower", ax=ax,
    )

    if cbar:
        cmap_hacked = psm.cm.bivariate.TransparencyBivariateColormap(
            cmap.reversed(),
            transmap=lambda x: 1
        )
        cax = psm.add_bivariate_colorbox(
            cmap_hacked,
            norm1,
            norm2,
            # xlabel="Fraction of vessels publicly tracked per $km^2$",
            xlabel="",
            ylabel="",
            xformat="{x:.0%}",
            aspect_ratio=50,  # lower = thicker
            width=0.45,
            loc=(0.36, -0.02),
            # fontsize=20,
            fontsize=FONT,
            ax=ax,
        )
        cax.minorticks_off()
        cax.get_yaxis().set_visible(False)
        [spine.set_visible(False) for key, spine in cax.spines.items()]
        cax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        cax.tick_params(labelsize=FONT)

        cax.text(
            0.5,
            0.5,
            "Fraction of vessels publicly tracked per $km^2$",  # cbar top
            ha="center",
            va="bottom",
            rotation=0,
            # fontsize=20,
            fontsize=FONT,
            color="0.1",
        )

    ax.axis("off")


# %%
scale = 5

# %%
# Load fishing data
# df = pd.read_csv('../data/raster_10th_degree_v20230217.csv.zip')
# df = pd.read_csv('../data/raster_5th_degree_v20230218.csv.zip')
df = pd.read_feather('../data/raster_5th_degree.feather')
df_bars = pd.read_csv('../data/vessels_bycontinent_v20230803.csv')
sorted(df_bars.columns)

# %%
df.head()

# %%
try:
    df = df.rename({
        'matched_fishing': 'ais_fishing',
        'matched_nonfishing': "ais_nonfishing",
        'unmatched_nonfishing': 'dark_nonfishing'
    }, axis='columns', errors="raise")
except:
    pass

df_bars = df_bars.rename({
    'AIS fishing': 'ais_fishing',
    'dark fishing': 'dark_fishing',
    'AIS non-fishing': "ais_nonfishing",
    'dark non-fishing': 'dark_nonfishing'
}, axis='columns', errors="raise")

df = df.rename(str.lower, axis='columns')
df_bars = df_bars.rename(str.lower, axis='columns')

# %%
df_bars

# %%
df["tot_fishing"] = df["dark_fishing"] + df["ais_fishing"]
df["tot_nonfishing"] = df["dark_nonfishing"] + df["ais_nonfishing"]


# %%
fishing_total = psm.rasters.df2raster(
    df,
    "lon_index",
    "lat_index",
    "tot_fishing",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

dark_fishing = psm.rasters.df2raster(
    df,
    "lon_index",
    "lat_index",
    "dark_fishing",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

fishing_ratio = dark_fishing / fishing_total

# %%
nonfishing_total = psm.rasters.df2raster(
    df,
    "lon_index",
    "lat_index",
    "tot_nonfishing",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

dark_nonfishing = psm.rasters.df2raster(
    df,
    "lon_index",
    "lat_index",
    "dark_nonfishing",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

nonfishing_ratio = dark_nonfishing / nonfishing_total


# %%
def add_labels(ind, x, labels, ax, color='0.5'):
    for i, label in zip(ind, labels):
        ax.text(
            x,
            i,
            label,
            ha="right",
            va="center",
            rotation=0,
            # fontsize=18,
            fontsize=FONT,
            color=color,
        )


def add_numbers(ind, size, labels, ax, pad=120, color="0.5"):
    for i, s, label in zip(ind, size, labels):
        ax.text(
            s + pad,
            i,
            label,
            ha="left",
            va="center",
            rotation=0,
            # fontsize=18,
            fontsize=FONT,
            color=color,
        )


def add_description(text, ax):
    ax.text(
        0.0,
        3.0,
        text,
        ha="left",
        va="center",
        rotation=0,
        # fontsize=18,
        fontsize=FONT,
        color="0.1",
    )

    
def add_chart(
    left,
    bottom,
    width,
    height,
    fig,
    a,
    b,
    y_label=-500,
    p_label=500,
    title="",
    labels=None,
):

    iax = fig.add_axes([left, bottom, width, height])

    labels = np.array(labels)
    colors = [blue, red]

    ind = np.array([0, 0.5, 1, 1.5, 2, 2.5])

    tot = a + b
    # i_sort = np.argsort(tot)
    # labels = labels[i_sort]
    # tot = tot[i_sort]
    # a = a[i_sort]
    # b = b[i_sort]
    perc = 100 * a/tot

    iax.barh(
        y=ind, height=0.22, width=a, align="center", color=colors[1], alpha=0.8
    )
    iax.barh(
        y=ind,
        height=0.22,
        width=b,
        align="center",
        color=colors[0],
        left=a,
        alpha=0.8,
    )

    labels1 = [f"{s:.1f}k" for s in tot / 1000]  # totals
    labels2 = [f"{p:.0f}%" for p in perc]        # percents
    add_labels(ind, y_label, labels, iax)
    add_numbers(ind, tot, labels1, iax, p_label)
    # add_numbers(ind, tot, labels2, iax, p_label + 3000, color=colors[1])
    add_description(title, iax)

    iax.axis("off")


def add_title(title, ax):
    ax.text(
        0.95,
        0.92,
        title,
        ha="right",
        va="center",
        # fontsize=20,
        fontsize=FONT,
        color="0.1",
        transform=ax.transAxes,
    )


def add_legend(ax):
    ax.text(
        0.90,
        0.73,
        "2017-2021",
        ha="right",
        va="center",
        rotation=0,
        # fontsize=18,
        fontsize=FONT,
        color='0.5',
    )
    ax.text(
        0.90,
        0.61,
        "Not publicly tracked",
        ha="right",
        va="center",
        rotation=0,
        # fontsize=18,
        fontsize=FONT,
        color=red,
    )
    ax.text(
        0.90,
        0.49,
        "Publicly tracked",
        ha="right",
        va="center",
        rotation=0,
        # fontsize=18,
        fontsize=FONT,
        color=blue,
    )
    
    
def add_percents(percents, ax, k=2):
    for i, (x,y,z) in enumerate(percents):
        ax.text(
            x,
            y,
            f'{z}%',
            ha="center",
            va="center",
            rotation=0,
            # fontsize=20,
            fontsize=FONT,
            color=blue,
            # transform=ax.transAxes,
        )
        if i == k:
            ax.text(
                x,
                y,
                '\n\n\n\npublicly\ntracked',
                ha="center",
                va="center",
                rotation=0,
                # fontsize=18,
                fontsize=FONT,
                color=blue,
                # transform=ax.transAxes,
            )
            

def get_geometry(fcsv):
    # saved from the notebook CreateStudyArea.ipynb
    df_shapes = pd.read_csv(fcsv)
    # it ended up saving as a wkt... need to turn it back into geometries
    df_shapes["study_area"] = df_shapes.study_area.apply(
        lambda x: shapely.wkt.loads(x)
    )
    df_shapes["study_area_02"] = df_shapes.study_area_02.apply(
        lambda x: shapely.wkt.loads(x)
    )
    df_shapes["study_area_05"] = df_shapes.study_area_05.apply(
        lambda x: shapely.wkt.loads(x)
    )
    return df_shapes


def add_geometry(df_shapes, ax, linewidth=0.01, edgecolor='0.9'):
    ax.add_geometries(
        df_shapes.study_area_05,
        crs=psm.identity,
        facecolor="None",
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

            
def mouse_event(event):
    print('{}, {}'.format(round(event.xdata, 0), round(event.ydata, 0)))


# %%

# %%
# # %matplotlib qt

SAVE = True

plt.rcParams["figure.autolayout"] = True

scl = 0.466666667

fig = plt.figure(figsize=(7, 6 * 3 * scl), constrained_layout=False)
# cid = fig.canvas.mpl_connect('button_press_event', mouse_event)

grid = GridSpec(5, 1)
sub1 = grid[0:2, 0]
sub2 = grid[2:4, 0]
sub3 = grid[4, 0]

prj = cartopy.crs.Robinson(central_longitude=0, globe=None)
# extent = (-145, 175, -62, 90)
extent = (-132, 172, -62, 90)

with psm.context(psm.styles.light):

    # ===== Fishing ===== #

    ax1 = psm.create_map(subplot=sub1, projection=prj)
    ax1.set_extent(extent, crs=psm.identity)
    ax1.axis("off")

    map_bivariate(
        fishing_total,
        fishing_ratio,
        "Fishing Activity",
        vmax=1,
        a_vmin=0.00005,
        a_vmax=0.01,
        eez_linewidth=0.4,
        eez_color="#5c5c5c",
        cmap=mycmap7.reversed(),
        ax=ax1,
        psm=psm,
        cbar=True,
    )
    
    percents = [
        (8265583, 4003494, 22),  # Asia
        (2759558, 5863734, 61),  # Europe
        (-8743407, 5421615, 17),  # N America
        (1997329, 1289603, 22),  # Africa
        (-5321280, -1387109, 23), # S America
        (12365359, -2674643, 25),  # Australia
    ]
    add_percents(percents, ax1)
    
    add_geometry(get_geometry('../data/study_area.csv.zip'), ax1)
    
    add_title("Industrial Fishing Vessels", ax1)
    
    ax1.text(.01, .98, 'a', fontsize=FONT+1, weight='bold', ha='left', va='top', transform=ax1.transAxes)
    plt.draw()
    
    # ===== Non-fishing ===== #

    ax2 = psm.create_map(subplot=sub2, projection=prj)
    ax2.set_extent(extent, crs=psm.identity)
    ax2.axis("off")

    map_bivariate(
        nonfishing_total,
        nonfishing_ratio,
        "Non-Fishing Activity",
        vmax=1,
        a_vmin=0.00005,
        a_vmax=0.01,
        eez_linewidth=0.4,
        eez_color="#5c5c5c",
        cmap=mycmap7.reversed(),
        ax=ax2,
        psm=psm,
        cbar=False,
    )
    
    percents = [
        (8265583, 4003494, 69),  # Asia
        (2759558, 5863734, 93),  # Europe
        (-8743407, 5421615, 81),  # N America
        (1997329, 1289603, 85),  # Africa
        (-5321280, -1387109, 82), # S America
        (12365359, -2674643, 83),  # Australia
    ]
    add_percents(percents, ax2)

    add_geometry(get_geometry('../data/study_area.csv.zip'), ax2)
    
    add_title("Transport and Energy Vessels", ax2)
    
    ax2.text(.01, .98, 'b', fontsize=FONT+1, weight='bold', ha='left', va='top', transform=ax2.transAxes)
    plt.draw()

    # ===== Bar charts ===== #
    
    df_bars = df_bars.replace(to_replace={
        'North America': 'N America',
        'South America': 'S America',
    })
    
    df_fish = df_bars.sort_values(by=['tot_fishing'], ascending=True)
    df_nonf = df_bars.sort_values(by=['tot_nonfishing'], ascending=True)
    
    a = df_fish.dark_fishing
    b = df_fish.ais_fishing
    c = df_nonf.dark_nonfishing
    d = df_nonf.ais_nonfishing
    
    ax3 = fig.add_subplot(sub3)
    ax3.axis("off")

    left, bottom, width, height = [0.17, 0.07, 0.28, 0.12]
    title = "Industrial Fishing (num. vessels)"
    add_chart(left, bottom, width, height, fig, a, b, -600, 650, title, df_fish.continent)

    left, bottom, width, height = [0.58, 0.07, 0.28, 0.12]
    title = "Transport and Energy (num. vessels)"
    add_chart(left, bottom, width, height, fig, c, d, -600, 650, title, df_nonf.continent)

    add_legend(ax3)
    
    ax3.text(0.09, 1.3, 'c', fontsize=FONT+1, weight='bold', ha='left', va='top', transform=ax3.transAxes)
    plt.draw()
    
    # ===== Percentages ===== #
    
    print(df_fish.continent[::-1])
    
    print('Dark Fishing')
    print(100 - (100 * b / (a + b)).round(0)[::-1])
    
    print('Dark NonFishing')
    print(100 - (100 * d / (c + d)).round(0)[::-1])
    
    print(df_fish.tot_fishing[::-1] / df_fish.tot_fishing.sum())

if SAVE:
    fig.canvas.draw() 
    ax1.draw(fig.canvas.renderer)
    ax2.draw(fig.canvas.renderer)
    ax3.draw(fig.canvas.renderer)
    plt.savefig(
        "Figure_1.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
        dpi='figure',
    )

# %%
