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
# DATA:
# - raster_5th_degree_v20230218.csv.zip
# - vessels_bycontinent_v20230217.csv

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
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta

# %matplotlib inline
plt.rcParams["savefig.dpi"] = 600

# %%
# use the standard for eliminating ice locations.
from prj_global_sar_analysis.eliminate_ice_string import eliminate_ice_string

eliminated_locations = eliminate_ice_string()


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


palette1 = [
    "#d7191c",
    "#fdae61",
    "#ffffbf",
    "#abd9e9",
    "#2c7bb6",
]

red, blue = "#ca0020", "#0571b0"  # original
# red, blue = '#E43026', '#2180B7'  # guppy

# Red-Blue
palette2 = [
    "#ca0020",
    # "#f4a582",
    # "#f7f7f7",  # white
    # "#92c5de",
    "#0571b0",
]

palette3 = [
    "#ca0020",
    "#f4a582",
    "#ffffbf",  # yellow
    "#92c5de",
    "#0571b0",
]

palette4 = [
    "#0571b0",
    "#92c5de",
    "#f4a582",
    "#ca0020",
]

palette5 = [
    "#2b83ba",
    # "#6C3B58",  # dark purple
    "#7b3294",  # light purple
    "#d7191c",
]

palette6 = [
    "#ca0020",  # red
    "#ffffbf",  # yellow
    "#0571b0",  # blue
]

# guppy
palette7 = [
    "#F77539",
    "#DE0026",
    "#44005F",
    "#2954E1",
    "#24BDB2",
]

mycmap1 = get_continuous_cmap(palette1)
mycmap2 = get_continuous_cmap(palette2)
mycmap3 = get_continuous_cmap(palette3)
mycmap4 = get_continuous_cmap(palette4)
mycmap5 = get_continuous_cmap(palette5)
mycmap6 = get_continuous_cmap(palette6)
mycmap7 = get_continuous_cmap(palette7)


# %%
def piecewise_constant_color_map(colors, name="pccm"):
    """colors is list[tuple(float, float, float)]"""
    breaks = np.linspace(0, 1.0, len(colors) + 1, endpoint=True)
    arg = {"red": [], "green": [], "blue": []}
    last_clr = colors[0]
    colors = colors + [colors[-1]]
    for i, clr in enumerate(colors):
        arg["red"].append((breaks[i], last_clr[0], clr[0]))
        arg["green"].append((breaks[i], last_clr[1], clr[1]))
        arg["blue"].append((breaks[i], last_clr[2], clr[2]))
        last_clr = clr
    return mpcolors.LinearSegmentedColormap(name, arg)


rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in palette7]
mycmap_discrete = piecewise_constant_color_map(rgb_list)


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

    cmap_bi = psm.cm.bivariate.TransparencyBivariateColormap(cmap)
    # ax = psm.create_map(ax)
    # psm.add_land(ax)
    # psm.add_eezs()
    # psm.add_eezs(ax, edgecolor=eez_color, linewidth=eez_linewidth)
    norm1 = mpcolors.Normalize(vmin=0.0, vmax=vmax, clip=True)
    norm2 = mpcolors.LogNorm(vmin=a_vmin, vmax=a_vmax, clip=True)

    img = psm.add_bivariate_raster(
        grid_ratio, grid_total, cmap_bi, norm1, norm2, origin="lower", ax=ax
    )

    if cbar:
        cmap_hacked = psm.cm.bivariate.TransparencyBivariateColormap(
            cmap.reversed(),
            transmap=lambda x: 1
            # psm.cm.bivariate.orange_blue, transmap=lambda x: 1
        )
        cax = psm.add_bivariate_colorbox(
            cmap_hacked,
            norm1,
            norm2,
            xlabel="Fraction of vessels matched to AIS per $km^2$",
            ylabel="",
            xformat="{x:.0%}",
            #  yformat="{x:.2f}",
            aspect_ratio=50,  # lower = thicker
            width=0.40,
            loc=(0.4, 0.04),
            fontsize=16,
            ax=ax,
        )
        cax.minorticks_off()
        cax.get_yaxis().set_visible(False)
        [spine.set_visible(False) for key, spine in cax.spines.items()]
        cax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        cax.tick_params(labelsize=14)

        cax.text(
            0.5,
            0.5,
            # "Add map description here",
            "",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=14,
            color="0.6",
        )

    # elif cbar:
    #     psm.add_top_labeled_colorbar(
    #     img,
    #     left_label=r"$\longleftarrow$ less matched",
    #     # center_label=r"AIS$\leftrightarrow$registries",
    #     center_label=r"",
    #     right_label=r"more matched $\longrightarrow$",
    # )

    # ax.set_title(title, x=0.8, y=0.962, pad=-14, ha="right", fontsize=16, color="0.0")

    ax.axis("off")


# %%
scale = 5

# %%
# Load fishing data
# df = pd.read_csv('../data/raster_10th_degree_v20230217.csv.zip')
df = pd.read_csv('../data/raster_5th_degree_v20230218.csv.zip')
df_bars = pd.read_csv('../data/vessels_bycontinent_v20230217.csv')

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
# df['country'] = df.eez_iso3.apply(get_country)
# df['continent'] = df.eez_iso3.apply(get_continent)
# df["AIS fishing"] = df.matched_fishing + df.matched_unknown_likelyfish
# df["AIS non-fishing"] = (df.matched_nonfishing + df.matched_unknown_likelynonfish)
# df["dark fishing"] = df.unmatched_fishing_prob
# df["dark non-fishing"] = df.unmatched_nonfishing_prob
# df["dark fishing t"] = df.unmatched_fishing_t
# df["dark non-fishing t"] = df.unmatched_nonfishing_t
# df["tot_fishing"] = df["dark fishing"] + df["AIS fishing"]

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
def add_labels(ind, x, labels, ax):
    for i, label in zip(ind, labels):
        ax.text(
            x,
            i,
            label,
            ha="right",
            va="center",
            rotation=0,
            fontsize=16,
            color="0.5",
        )


def add_numbers(ind, size, labels, ax, pad=120):
    for i, s, label in zip(ind, size, labels):
        ax.text(
            s + pad,
            i,
            label,
            ha="left",
            va="center",
            rotation=0,
            fontsize=16,
            color="0.5",
        )


def add_description(text, ax):
    ax.text(
        0.0,
        3.0,
        text,
        ha="left",
        va="center",
        rotation=0,
        fontsize=16,
        color="0.0",
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
    colors = ["#218AB2", "#F05A2B"]  # turquose3

    ind = np.array([0, 0.5, 1, 1.5, 2, 2.5])

    tot = a + b
    i_sort = np.argsort(tot)
    labels = labels[i_sort]
    tot = tot[i_sort]
    a = a[i_sort]
    b = b[i_sort]

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

    labels1 = [f"{s:.1f}k" for s in tot / 1000]
    # labels2 = labels
    labels2 = [f"{s1} {s2}" for s1, s2 in zip(labels, labels1)]
    add_labels(ind, y_label, labels, iax)
    add_numbers(ind, tot, labels1, iax, p_label)
    add_description(title, iax)

    iax.axis("off")


def add_title(title, ax):
    ax.text(
        0.95,
        0.92,
        title,
        ha="right",
        va="center",
        fontsize=20,
        color="0.0",
        transform=ax.transAxes,
    )


def add_legend(ax):
    orange = "#F05A2B"
    blue = "#218AB2"
    ax.text(
        0.90,
        0.60,
        "2017-2021",
        ha="right",
        va="center",
        rotation=0,
        fontsize=16,
        color='0.5',
    )
    ax.text(
        0.90,
        0.50,
        "Dark vessels",
        ha="right",
        va="center",
        rotation=0,
        fontsize=16,
        color=orange,
    )
    ax.text(
        0.90,
        0.40,
        "Publicly tracked",
        ha="right",
        va="center",
        rotation=0,
        fontsize=16,
        color=blue,
    )


# %%
SAVE = True

# LambertAzimuthalEqualArea
plt.rcParams["figure.autolayout"] = True

fig = plt.figure(figsize=(15, 6 * 3), constrained_layout=False)

grid = GridSpec(5, 1)
sub1 = grid[0:2, 0]
sub2 = grid[2:4, 0]
sub3 = grid[4, 0]

prj = cartopy.crs.Robinson(central_longitude=0, globe=None)
extent = (-145, 175, -62, 90)

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
        # cmap=mycmap2.reversed(),
        cmap=mycmap7.reversed(),
        ax=ax1,
        psm=psm,
        cbar=True,
    )

    add_title("Industrial Fishing", ax1)

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
        # cmap=mycmap2.reversed(),
        cmap=mycmap7.reversed(),
        ax=ax2,
        psm=psm,
        cbar=False,
    )

    add_title("Transport and Energy", ax2)

    # ===== Bar charts ===== #
    
    # Data for barch charts
    # a = np.array([1690, 20608, 460, 1238, 1969, 874])   # dark fish
    # b = np.array([361, 3228, 116, 1615, 418, 330])      # ais fish
    # c = np.array([256, 5670, 41, 385, 237, 96])         # dark nonf
    # d = np.array([2095, 13722, 643, 4374, 2055, 1093])  # ais nonf
    
    df_bars['continents'] = [
        'Asia',
        'Europe',
        'N America',
        'Africa',
        'S America',
        'Australia'
     ]
    
    df_fish = df_bars.sort_values(by=['tot_fishing'], ascending=False)
    df_nonf = df_bars.sort_values(by=['tot_nonfishing'], ascending=False)
    
    a = df_fish.dark_fishing
    b = df_fish.ais_fishing
    c = df_nonf.dark_nonfishing
    d = df_nonf.ais_nonfishing
    
    ax3 = fig.add_subplot(sub3)
    ax3.axis("off")

    left, bottom, width, height = [0.17, 0.07, 0.28, 0.11]
    title = "Industrial Fishing (num. vessels)"
    add_chart(left, bottom, width, height, fig, a, b, -600, 650, title, df_fish.continents)

    left, bottom, width, height = [0.58, 0.07, 0.28, 0.11]
    title = "Transport and Energy (num. vessels)"
    add_chart(left, bottom, width, height, fig, c, d, -600, 650, title, df_nonf.continents)

    add_legend(ax3)

if SAVE:
    plt.savefig("figures/fig1v2b_5th.png", bbox_inches="tight", pad_inches=0, dpi=300)

# %%
