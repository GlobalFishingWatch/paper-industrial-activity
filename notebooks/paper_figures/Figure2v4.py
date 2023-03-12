# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: py:percent
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

# %% [markdown]
# # Make Figure 2
#
#

# %%
# %matplotlib inline

import cartopy
import cartopy.crs as ccrs
import matplotlib.colors as mpcolors
from matplotlib import gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import pandas as pd
import geopandas as gpd
import proplot as pplt
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
import pyperclip

# for making the outline of the study area
import shapely


# %% [markdown]
# ## Color configurations

# %%
# colors for matched and unmatched in the plots
matched_color = "#07A9FD"
dark_color = "#FD7F0C"


# the bivariate raster cmaps
cmap = psm.cm.bivariate.TransparencyBivariateColormap(
    psm.cm.bivariate.blue_orange
)
norm1 = mpcolors.Normalize(vmin=0.0, vmax=1, clip=True)
norm2 = mpcolors.LogNorm(vmin=0.0001, vmax=0.01, clip=True)

# the outlines of each plot
axcolor = "#cccccc"

# callout box color
zoom_box_color = "black"

# study area
study_linewidth = 0.25
study_color = "#303030"

# %% [markdown]
# ## Regions and countries for the plots

# %%
med_bbox = -9.4, 26.7, 35, 49
tunisa_bbox = 10.12, 32.9, 16.29, 38.76
sasia_bbox = 63.9, 4.7, 107.1, 26.6
bangla_bbox = 87.72, 17.9, 93.3, 23.62
easia_bbox = 115.2, 30.8, 145.6, 45.9
nkorea_bbox = 122.08, 37.71, 127.16, 40

# Get all the bboxes we want for scatterplots
# This is used to limit the dataframe for positions
all_scatterplot_bboxes = [tunisa_bbox, bangla_bbox, nkorea_bbox]

locations_med = {
    "ITA": (10.338905, 45.832189),
    "FRA": (2.767959, 45.147904),
    "ESP": (-3.552294398649515, 40.265859858535954),
    "MAR": (-5.408427865243035, 32.68240583159011),
    "DZA": (3.073398546953262, 33.97798689022415),
    "TUN": (9.50024882458396, 33.0542129837118),
    "LBY": (16.88867205636247, 30.032789663517814),
    "EGY": (29.153251634021395, 29.207819427587005),
    "GRC": (21.60050845673003, 39.77129040227394),
    "PRT": (-8.232150, 39.272230),
    "HRV": (15.465627694810284, 45.29233070976646),
    "MNE": (19.098109672561826, 42.767909907539384),
    "ALB": (20.10044882969955, 40.92540373891045),
    "TUR": (32.0427974532703, 39.03489621387328),
    "ISR": (35.00652652438128, 31.64267755459708),
    "SYR": (36.84875395648638, 35.33947051048809),
    "LBN": (35.91298004925963, 33.98675191752204),
    "CYP": (33.121071, 35.013657),
}

locations_sasia = {
    "IND": (78.22419945360045, 22.093336885517907),
    "BGD": (90.10390386747254, 23.923835206284416),
    "PAK": (67.5599283092789, 26.10421733108435),
    "MMR": (95.40401774333814, 20.739555737581448),
    "LKA": (80.86197522125954, 7.432377768806482),
    "THA": (100.8223467483026, 17.009976625100965),
}

locations_easia = {
    "CHN": (118.74702919866394, 43.34593982311163),
    "JPN": (138.58017169423337, 36.15135425770859),
    "PRK": (126.6324087319412, 40.62319551106906),
    "KOR": (127.91568697603923, 36.40107172420694),
}

# %% [markdown]
# ## Read in all detections and apply categories and colors

# %%
# Read in GIANT csv of all detections, created by DownloadAllDetections.ipynb
# this is a huge file... >1.5gb
# in this notebook, df is the master dataframe of all the dections.
# we will filter its size down below.
df = pd.read_csv("../data/all_detections.csv.zip")

# %%
# 23.6 million rows!
len(df) / 1e6

# %%
# limit these detections to only places that are inside our bounding boxes,
# plus a degree constant which we will set to 3 degrees.

ds = []
a = 3
for bbox in all_scatterplot_bboxes:
    min_lon, min_lat, max_lon, max_lat = bbox
    d = df[
        (df.lon > min_lon - a)
        & (df.lon < max_lon + a)
        & (df.lat > min_lat - a)
        & (df.lat < max_lat + a)
    ]
    ds.append(d)

df = pd.concat(ds)
df = df.drop_duplicates()
len(df) / 1e6
# down toi 3.3 million rows

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
# color is a hex,
# color2 is a tuple rgb
# we are basing this on category_rand


def get_color(row):
    if row.category_rand in ("matched_fishing", "matched_nonfishing"):
        return matched_color

    else:
        return dark_color


df["color"] = df.apply(get_color, axis=1)

# make a column that has color saved as a tuple... apparently ax.scatter needs a list of tuples,
# not a a list of hexes... go figure.
df["color2"] = df.color.apply(
    lambda h: tuple(int(h[1:][i : i + 2], 16) / 256 for i in (0, 2, 4))
)

# %% [markdown]
# ## Read in statistics by EEZ for the pie charts

# %%
# read in activity by EEZ, which is generated by another notebook
df_eez = pd.read_csv("../data/activity_by_eez.csv")

# %% [markdown]
# ## Read in dataframe that produces the raster
#
# This dataframe is produced by CreateDarkActivityRasterDataframe.ipynb

# %%
dfr = pd.read_csv("../data/raster_20th_degree.csv.zip")

# %%
scale = 20

# %%
fishing_total = psm.rasters.df2raster(
    dfr,
    "lon_index",
    "lat_index",
    "tot_fishing",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

dark_fishing = psm.rasters.df2raster(
    dfr,
    "lon_index",
    "lat_index",
    "dark_fishing",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

fishing_ratio = dark_fishing / fishing_total

# %% [markdown]
# ## Load study area polygons

# %%
# saved from the notebook CreateStudyArea.ipynb
df_shapes = pd.read_csv("../data/study_area.csv.zip")

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


# %% [markdown]
# ## plotting helping functions

# %%
def get_projection(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    the_extent = [min_lon, max_lon, min_lat, max_lat]
    the_center = [min_lon / 2 + max_lon / 2, min_lat / 2 + max_lat / 2]
    projection = cartopy.crs.LambertAzimuthalEqualArea(
        central_latitude=the_center[1], central_longitude=the_center[0]
    )
    return projection


def get_map_extent(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    map_extent = (min_lon, max_lon, min_lat, max_lat)
    return map_extent


def add_pies(ax, projection, locations, df_eez):

    pie_scale = 0.2  # adjust the base size of the pie charts

    for eez in locations:
        de = df_eez[df_eez.ISO_TER1 == eez]
        de = de.groupby("ISO_TER1").sum()
        lon, lat = locations[eez]
        ilon, ilat = projection.transform_point(lon, lat, ccrs.PlateCarree())

        width = pie_scale * (de.fishing.sum() / 300) ** 0.5

        ax_sub = inset_axes(
            ax,
            width=width,
            height=width,
            loc=10,
            bbox_to_anchor=(ilon, ilat),
            bbox_transform=ax.transData,
            borderpad=0,
        )

        wedges, texts = ax_sub.pie(
            [de.matched_fishing.values[0], de.dark_fishing.values[0]],
            colors=[matched_color, dark_color],
            wedgeprops={"alpha": 0.1},
        )

        ax_sub.set_aspect("equal")


def scatter_points(ax, bbox, size, a=2, years=None):

    min_lon, min_lat, max_lon, max_lat = bbox
    d = df[
        (df.lon > min_lon - a)
        & (df.lon < max_lon + a)
        & (df.lat > min_lat - a)
        & (df.lat < max_lat + a)
    ]
    d2 = d[d.category_rand.isin([f"matched_fishing", f"dark_fishing"])]
    if years is not None:
        d2 = d2[d2.year.isin(years)]

    sct = ax.scatter(
        d2.lon,
        d2.lat,
        c=d2.color2,
        alpha=0.5,
        s=size,
        edgecolors="none",
        rasterized=True,
        transform=psm.identity,
    )


def set_axcolor(ax, color):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)


def plot_callout_box(ax, bbox):
    # add call out box
    lon1, lat1, lon2, lat2 = bbox
    ax.plot(
        [lon1, lon1, lon2, lon2, lon1],
        [lat1, lat2, lat2, lat1, lat1],
        transform=psm.identity,
        color=zoom_box_color,
        linewidth=0.5,
    )


def add_raster(ax):
    psm.add_bivariate_raster(
        fishing_ratio,
        fishing_total,
        cmap,
        norm1,
        norm2,
        origin="lower",
        ax=ax,
    )


# %% [markdown]
# ## Make figure 2

# %%
# create a figure with 3 rows and 2 columns, but split
# the bottom right into two. 7 inches by 8, so it will
# fill up the majority of a page in the submission

SAVE = False

fig = plt.figure(figsize=(7, 8))

# specify the layout of the subplots using gridspec
gs = gridspec.GridSpec(
    nrows=4, ncols=2, width_ratios=[2, 1], height_ratios=[2, 2, 1, 1]
)

landcolor = '0.95'

# Mediterranean raster and pies
projection = get_projection(med_bbox)
ax = fig.add_subplot(gs[0, 0], projection=projection)
set_axcolor(ax, axcolor)
ax.set_extent(get_map_extent(med_bbox))
psm.add_land(ax, color=landcolor)
psm.add_countries(ax)
# psm.add_eezs(ax)
add_pies(ax, projection, locations_med, df_eez)
add_raster(ax)
# add call out box
plot_callout_box(ax, tunisa_bbox)
# ax.add_geometries(
#     df_shapes.study_area_05,
#     crs=psm.identity,
#     facecolor="0.95",
#     edgecolor=study_color,
#     linewidth=study_linewidth,
#     zorder=0,
# )


# Tunisia and S. Italy scatter
projection = get_projection(tunisa_bbox)
ax = fig.add_subplot(gs[0, 1], projection=projection)
set_axcolor(ax, axcolor)
ax.set_extent(get_map_extent(tunisa_bbox))
psm.add_land(ax, color=landcolor)
# psm.add_eezs(ax)
scatter_points(ax, tunisa_bbox, size=0.05)
# ax.add_geometries(
#     df_shapes.study_area_02,
#     crs=psm.identity,
#     facecolor="0.95",
#     edgecolor=study_color,
#     linewidth=study_linewidth,
#     zorder=0,
# )


# South Asia raster and pies
projection = get_projection(sasia_bbox)
ax = fig.add_subplot(gs[1, 0], projection=projection)
set_axcolor(ax, axcolor)
ax.set_extent(get_map_extent(sasia_bbox))
psm.add_land(ax, color=landcolor)
psm.add_countries(ax)
# psm.add_eezs(ax)
add_pies(ax, projection, locations_sasia, df_eez)
add_raster(ax)
# ax.add_geometries(
#     df_shapes.study_area_05,
#     crs=psm.identity,
#     facecolor="0.95",
#     edgecolor=study_color,
#     linewidth=study_linewidth,
#     zorder=0,
# )


# add call out box
plot_callout_box(ax, bangla_bbox)
lon1, lat1, lon2, lat2 = bangla_bbox


# Bangla scatter
projection = get_projection(bangla_bbox)
ax = fig.add_subplot(gs[1, 1], projection=projection)
set_axcolor(ax, axcolor)
ax.set_extent(get_map_extent(bangla_bbox))
psm.add_land(ax, color=landcolor)
# psm.add_eezs(ax)
scatter_points(ax, bangla_bbox, size=0.05)
# ax.add_geometries(
#     df_shapes.study_area_02,
#     crs=psm.identity,
#     facecolor="0.95",
#     edgecolor=study_color,
#     linewidth=study_linewidth,
#     zorder=0,
# )


# East Asia Plot
projection = get_projection(easia_bbox)
ax = fig.add_subplot(gs[2:4, 0], projection=projection)
set_axcolor(ax, axcolor)
ax.set_extent(get_map_extent(easia_bbox))
psm.add_land(ax, color=landcolor)
psm.add_countries(ax)
# psm.add_eezs(ax)
add_pies(ax, projection, locations_easia, df_eez)
add_raster(ax)
# add call out box
plot_callout_box(ax, nkorea_bbox)
# ax.add_geometries(
#     df_shapes.study_area_05,
#     crs=psm.identity,
#     facecolor="0.95",
#     edgecolor=study_color,
#     linewidth=study_linewidth,
#     zorder=0,
# )


# Nkorea 2018,2019 scatter
projection = get_projection(nkorea_bbox)
ax = fig.add_subplot(gs[2, 1], projection=projection)
set_axcolor(ax, axcolor)
ax.set_extent(get_map_extent(nkorea_bbox))
psm.add_land(ax, color=landcolor)
# psm.add_eezs(ax)
scatter_points(ax, nkorea_bbox, years=[2018, 2019], size=0.05)
# ax.add_geometries(
#     df_shapes.study_area_02,
#     crs=psm.identity,
#     facecolor="0.95",
#     edgecolor=study_color,
#     linewidth=study_linewidth,
#     zorder=0,
# )


# Nkorea 2020,2021 scatter
ax = fig.add_subplot(gs[3, 1], projection=projection)
set_axcolor(ax, axcolor)
ax.set_extent(get_map_extent(nkorea_bbox))
psm.add_land(ax, color=landcolor)
# psm.add_eezs(ax)
scatter_points(ax, nkorea_bbox, years=[2020, 2021], size=0.05)
# ax.add_geometries(
#     df_shapes.study_area_02,
#     crs=psm.identity,
#     facecolor="0.95",
#     edgecolor=study_color,
#     linewidth=study_linewidth,
#     zorder=0,
# )


# make it tight
plt.subplots_adjust(wspace=0.02, hspace=0.02)
plt.tight_layout()

if SAVE:
    plt.savefig("figures/fig2v4b.png", dpi=300, bbox_inches="tight")
# %%

