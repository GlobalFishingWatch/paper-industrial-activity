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
# def get_lonlat_from_id(detect_ids):
#     lons = [float(d.split(";")[1]) for d in detect_ids]
#     lats = [float(d.split(";")[2]) for d in detect_ids]
#     return np.array(lons), np.array(lats)

# %% [markdown]
# ## Map global fishing activity

# %%
# q = """
# select 
#   detect_lat,
#   detect_lon,
#   detect_id,
#   length_m,
#   score>1e-2 as is_matched_to_ais
# from
#   proj_global_sar.detections_w_overpasses_v20220805
# where
#   -- the following is very restrictive on repeated objects
#   repeats_100m_180days_forward < 3 and
#   repeats_100m_180days_back < 3 and
#   repeats_100m_180days_center < 3
#   -- get rid of scenes where more than half the detections
#   -- are likely noise
#   and (scene_detections <=5 or scene_quality > .5)
#   and presence > .7
#   and extract(date from detect_timestamp) 
#      between "2017-01-01" and "2021-12-31"
# """

# %%
# # --- Query and save data --- #

# # df = pd.read_gbq(q)
# # df.to_feather('data/vessel_detections_2017_2021.feather')

# # --- Load vessel data --- #

# df = pd.read_feather("data/vessel_detections_2021.feather", use_threads=True)
# df = df.rename(columns={"detect_lon": "lon", "detect_lat": "lat"})

# df.head()

# %%
df = pd.read_feather("../../data/all_detections_matched_rand_2021.feather")

# %%
len(df[df.category_rand != "none"])/len(df)

# %%
# drop unknown vessels, which are less than 1/1000th of the total
df = df[df.category_rand != "none"]

# %%
df.head()

# %%
df["is_matched_to_ais"] = df.category_rand.isin(['matched_fishing','matched_nonfishing']).astype(int)


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


palette3 = ["#FF6855", "#E19664", "#C1C674", "#57E69B", "#00FFBB"]  # OrTq1
# palette2 = ['#FF705E', '#FF9046', '#CAD33D', '#46FF9E', '#00FFBB']  # OrTq2
# palette = ['#FF5DB2', '#FF9063', '#BEC35E', '#3EF35A', '#1EE9B3']  # PkTq
# palette3 = ['#FF5DB2', '#FF9063', '#3EF35A', '#1EE9B3']  # PkTq mod
# palette2 = ["#ff6854", "#FF705E", "#46FF9E", "#00FFBB"]  # OrTq2 mod
# palette2 = ['#FF705E', '#FF9046', '#CAD33D', '#21FFFF', '#21FFFF']  # OrBu
# palette3 = ['#FFFF0D', '#21FFFF']
# palette3 = ['#21FFFF', '#FF5DB2']


palette0 = ["#EA6E42", "#4493EC"]  # Orange-Blue
palette2 = [
    "#ca0020",
    "#f4a582",
    "#f7f7f7",
    "#92c5de",
    "#0571b0",
]  # Red-Blue
# palette = ["#7b3294","#c2a5cf","#f7f7f7","#a6dba0","#008837",] # Green-Purple

palette = [
    # '#bae4bc',
    "#7bccc4",
    "#43a2ca",
    "#0868ac",
    "#000000",
]

palette1 = [
    "#b2182b",
    # '#d6604d',
    # '#f4a582',
    # '#fddbc7',
    # '#f7f7f7',
    # '#d1e5f0',
    # '#92c5de',
    # '#4393c3',
    "#2166ac",
]

# palette1 = ['#1D8F64', '#DE0077']
# palette2 = ['#40A33A', '#843692']
palette4 = ["#921637", "#FFFFD9", "#12275E"]

mycmap0 = get_continuous_cmap(palette0, n=2)
mycmap1 = get_continuous_cmap(palette1, n=2)
mycmap2 = get_continuous_cmap(palette2)
mycmap4 = get_continuous_cmap(palette4)

# %%
df = df.fillna(0)

# %%
# (lon_center, lat_center, approximate_size in degrees)
# a = (2.8125, 52.29504228453735, 8.0)  # North Sea, UK-France
a = (6.3720703125, 55.83831352210821, 6)  # North Sea, Denmark 1
# a = (6.416015625, 55.27911529201561, 10)  # North Sea, Denmark 2
# b = (15.4248046875, 38.37611542403604, 16)  # Mediterranean Sea, Tunisia-Italy
b = (19.721441, 38.226403, 15)  # Mediterranean 2
c = (52.00927734375, 26.88288045572338, 6)  # Persian Gulf
# d = (101.4697265625, 7.1663003819031825, 20)  # Gulf of Thailand
d = (107.258931, 6.254869, 26)  # Indonesia
# d = (102.612364,13.560721, 20)  # Indonesia 2
# d = (104.785017,13.842161, 20)  # Indonesia 3
# d = (106.771440,13.992795, 22)  # Indonesia 4
# g = (14.765625, 69.50376519563686, 10)  # North Norway
# h = (-5.8447265625, 46.14939437647686, 8)  # Bay of Biscay 1
# i = (-5.8447265625, 46.70973594407157, 8)  # Bay of Biscay 2
# j = (-71.553955078125, 9.838979375579344, 0.8)  # Lake Maracaibo
e = (122.55273437499999, 35.47485808497102, 8)  # Bohai Sea, China
# e = (121.55273437499999, 37.47485808497102, 20)  # Bohai Sea, China
# e = (121.363714,32.815907, 12)  # Yellow Sea
f = (131.46240234375, 34.57895241036948, 7)  # Sea of Japan 2
g = (-90.90087890624999, 24.206889622398023, 11)  # Gulf of Mexico
# h = (-5.8447265625, 46.70973594407157, 8)  # Bay of Biscay 2
# h = (-15.617967,12.655499, 8.5)  # West Africa
h = (-16.150369, 13.950366, 10.5)  # West Africa 2
# l = (-92.988104,24.178952, 11.5)  # Gulf of Mexico
l = (-92.200000, 24.178952, 13)  # Gulf of Mexico
m = (-44.784018, -24.923780, 11)  # Southeast Brazil

# %%
SAVE = False

fig = plt.figure(figsize=(8, 9.5))  # set aspect ratio
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(8, 7)  # grid cells

gspecs = [
    (slice(0, 4), slice(0, 7)),
    (slice(4, 8), slice(0, 4)),
    (slice(4, 8), slice(4, 7)),
]

proj_info = [b, d, e]

count = 0

with psm.context(psm.styles.light):
    with psm.context({"text.color": "k"}):
        for g, p in zip(gspecs, proj_info):
            print(g)

            count += 1

            if count == 1:
                size = 0.06
            elif count == 2:
                # size = 0.15
                size = 0.02
            elif count == 3:
                # size = 0.3
                size = 0.02

            xc, yc, dx_x_2 = p
            dx = dx_x_2 / 2
            extent = (xc - dx, xc + dx, yc - dx, yc + dx)
            prj = cartopy.crs.LambertAzimuthalEqualArea(xc, yc)

            ax = psm.create_map(subplot=gs[g], projection=prj)

            buff = 10
            df1 = df[
                (df.lon > extent[0] - buff)
                & (df.lon < extent[1] + buff)
                & (df.lat > extent[2] - buff)
                & (df.lat < extent[3] + buff)
            ]

            df2 = df1[df1.is_matched_to_ais == True]
            df3 = df1[df1.is_matched_to_ais == False]

            sct = ax.scatter(
                df2.lon,
                df2.lat,
                c=df2.is_matched_to_ais,
                s=size,
                cmap=mycmap0,
                vmin=0,
                vmax=1,
                edgecolors="none",
                rasterized=True,
                transform=psm.identity,
                zorder=10,
            )
            sct = ax.scatter(
                df3.lon,
                df3.lat,
                c=df3.is_matched_to_ais,
                s=size,
                cmap=mycmap0,
                vmin=0,
                vmax=1,
                edgecolors="none",
                rasterized=True,
                transform=psm.identity,
                zorder=100,
            )

            psm.add_land(ax=ax)
            ax.set_extent(extent, crs=psm.identity)
            ax.set_adjustable("datalim")

            # --- Colorbar --- #

            if count == 2:

                cbar = psm.add_top_labeled_colorbar(
                    sct,
                    ax=ax,
                    fig=fig,
                    left_label="",
                    center_label="",
                    right_label="",
                    loc="bottom",
                    width=1.0,
                    height=0.02,
                    format=None,
                    vertical_space=0.03,
                    ticks=[0.2, 0.82],
                )
                cbar.set_xticklabels(
                    ("Unmatched (dark)", "Matched to AIS"), fontsize=10
                )
                cbar.invert_xaxis()

plt.subplots_adjust(
    left=0.1,
    bottom=0.1,
    right=0.9,
    top=0.9,
    wspace=0.13,
    hspace=0.1,
)

if SAVE:
    plt.savefig(
        "../../fig2_fishing_points.png", bbox_inches="tight", pad_inches=0.01, dpi=172
    )

# %%
df_raster = pd.read_feather("../../data/raster_20th_degree.feather")

# %%
df_raster.head()

# %%

# %%
scale = 20

fishing_total = psm.rasters.df2raster(
    df_raster,
    "lon_index",
    "lat_index",
    "tot_fishing",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

dark_fishing = psm.rasters.df2raster(
    df_raster,
    "lon_index",
    "lat_index",
    "dark_fishing",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

fishing_ratio = dark_fishing / fishing_total

# %%
vmax = 1
a_vmin = 0.00005
a_vmax = 0.01

grid_total = fishing_total
grid_ratio = fishing_ratio

cmap = psm.cm.bivariate.TransparencyBivariateColormap(
    psm.cm.bivariate.blue_orange
)
norm1 = mpcolors.Normalize(vmin=0.0, vmax=vmax, clip=True)
norm2 = mpcolors.LogNorm(vmin=a_vmin, vmax=a_vmax, clip=True)

fig = plt.figure(figsize=(8, 9.5))  # set aspect ratio
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(8, 7)  # grid cells

gspecs = [
    (slice(0, 4), slice(0, 7)),
    (slice(4, 8), slice(0, 4)),
    (slice(4, 8), slice(4, 7)),
]

proj_info = [b, l, h]

count = 0

with psm.context(psm.styles.light):
    with psm.context(
        {"text.color": "k", "pyseas.map.colorbarlabelfont": {"fontsize": 10}}
    ):
        for g, p in zip(gspecs, proj_info):
            print(g)

            count += 1

            xc, yc, dx_x_2 = p
            dx = dx_x_2 / 2
            extent = (xc - dx, xc + dx, yc - dx, yc + dx)
            prj = cartopy.crs.LambertAzimuthalEqualArea(xc, yc)

            ax = psm.create_map(subplot=gs[g], projection=prj)

            img = psm.add_bivariate_raster(
                grid_ratio,
                grid_total,
                cmap,
                norm1,
                norm2,
                origin="lower",
                ax=ax,
            )

            psm.add_land(ax=ax)
            ax.set_extent(extent, crs=psm.identity)
            ax.set_adjustable("datalim")

            if count == 2:

                cbar = psm.add_top_labeled_colorbar(
                    img,
                    ax=ax,
                    fig=fig,
                    left_label="less",
                    center_label="Fraction of dark fishing",
                    right_label="more",
                    loc="bottom",
                    width=1.0,
                    height=0.02,
                    format=None,
                    vertical_space=0.06,
                    ticks=[0, 0.25, 0.5, 0.75, 1],
                )
                cbar.invert_xaxis()
                cbar.tick_params(labelsize=8)
                cbar.minorticks_off()


plt.subplots_adjust(
    left=0.1,
    bottom=0.1,
    right=0.9,
    top=0.9,
    wspace=0.13,
    hspace=0.1,
)

plt.savefig("../../figures/fig2_fishing.png", bbox_inches="tight", pad_inches=0.01, dpi=172)

# %%
