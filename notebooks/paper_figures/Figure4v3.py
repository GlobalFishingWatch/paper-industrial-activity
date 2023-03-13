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
# DATA, latest tables:
#
#     proj_global_sar.infrastructure_reclassified_v20230222
#     proj_global_sar.infra_eez_time_series_v20230222
#     proj_global_sar.infra_global_time_series_v20230222
#     proj_global_sar.infra_poly_time_series_v20230222
#     proj_global_sar.infra_vessel_activity_oil_100th_degree_v20230222
#     roj_global_sar.infra_vessel_activity_oil_10th_degree_v20230222
#     proj_global_sar.infra_vessel_activity_wind_100th_degree_v20230222
#     proj_global_sar.infra_vessel_activity_wind_10th_degree_v20230222
#     proj_global_sar.offshore_infra_w_regions_v20230222
#     proj_global_sar.offshore_infra_reclassified_w_regions_v20230222

# %%
# import cartopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import pyseas.maps as psm
import skimage
import cartopy
from matplotlib.gridspec import GridSpec

# %matplotlib inline

# %%
# Local import
from color import *

import pycountry
from pycountry_convert import (
    country_alpha2_to_continent_code,
    country_name_to_country_alpha2,
)

# %%
continents = {
    "NA": "North America",
    "SA": "South America",
    "AS": "Asia",
    "OC": "Australia",
    "AF": "Africa",
    "EU": "Europe",
}


def get_continent(x):
    try:
        return continents[
            country_alpha2_to_continent_code(
                pycountry.countries.get(alpha_3=x).alpha_2
            )
        ]
    except:
        "None"


def get_country(x):
    try:
        return country_alpha2_to_country_name(
            pycountry.countries.get(alpha_3=x).alpha_2
        )
    except:
        "None"


# %%
# Load table / read data on vessel traffic
QUERY = True

tab_oil = 'infra_vessel_activity_oil_100th_degree_v20230222'
tab_wind = 'infra_vessel_activity_wind_100th_degree_v20230222'

if QUERY:
    qo = f"SELECT * FROM proj_global_sar.{tab_oil}"
    qw = f"SELECT * FROM proj_global_sar.{tab_wind}"
    df_oil = pd.read_gbq(qo, project_id="world-fishing-827")
    df_wind = pd.read_gbq(qw, project_id="world-fishing-827") 
    df_oil.to_feather(f"../../data/{tab_oil}.feather")
    df_wind.to_feather(f"../../data/{tab_wind}.feather")
else:
    df_oil = pd.read_feather(f"../../data/{tab_oil}.feather", use_threads=True)
    df_wind = pd.read_feather(f"../../data/{tab_wind}.feather", use_threads=True)

df_oil.head()

# %%
df_oil.groupby("best_class")["visit_hours_sum"].sum().sort_values(
    ascending=False
)

df_wind.groupby("best_class")["visit_hours_sum"].sum().sort_values(
    ascending=False
)

# %%
classes = {
    "supply_vessel": "supply",
    "tanker": "tanker",
    "tug": "tug",
    "dive_vessel": "dive",
    "cargo": "cargo",
    "passenger": "passenger",
    "seismic_vessel": "seismic",
    # 'trawlers': 'trawler',
    # 'fishing': 'fishing',
    # 'non_fishing': 'nonfishing',
}

dfs_oil = dict()
dfs_wind = dict()

for key, val in classes.items():
    dfs_oil[val] = df_oil[df_oil.best_class == key]
    dfs_wind[val] = df_wind[df_wind.best_class == key]


# %%
def print_vessels(dfs, classes):
    for val in classes.values():
        d = dfs[val]
        d = d[d.visit_hours_sum > 0]
        print(
            f"{val: <10}",
            f"{d.visit_hours_sum.min().round(4): >10}",
            f"{d.visit_hours_sum.max().round(1): >10}",
            f"{d.visit_hours_sum.count(): >10}",
        )


print("OIL")
print_vessels(dfs_oil, classes)
print("\nWIND")
print_vessels(dfs_wind, classes)

# %%
scale = 100

raster_oil = psm.rasters.df2raster(
    df_oil,
    "lon_index",
    "lat_index",
    "visit_hours_sum",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

raster_wind = psm.rasters.df2raster(
    df_wind,
    "lon_index",
    "lat_index",
    "visit_hours_sum",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

# %%
# Load infrastructure data
QUERY = True

tab_infra = "offshore_infra_reclassified_w_regions_v20230222"

if QUERY:
    q = f"SELECT * FROM proj_global_sar.{tab_infra}  -- single date"
    df2 = pd.read_gbq(q, project_id="world-fishing-827")
    df2.to_feather(f"../../data/{tab_infra}.feather")
else:
    df2 = pd.read_feather(f"../../data/{tab_infra}.feather", use_threads=True)

# Add columns for binning
df2["x_index"] = (df2.lon * 100).astype(int)
df2["y_index"] = (df2.lat * 100).astype(int)
df2["count"] = 1
df2.head()

# %%
print(df2.oil_region.unique())
print(df2.label.unique())

# %%
# Get infrastructure detections
oil = df2.loc[df2.label.isin(["oil", "probable_oil"])]
wind = df2.loc[df2.label.isin(["wind", "probable_wind"])]
other = df2.loc[df2.label.isin(["unknown", "possible_oil", "possible_wind"])]
mara = df2.loc[df2.label == "lake_maracaibo"]

# %%
len(other)

# %%
# Rasterize infrastructure

xyscale = 1  # N degree resolution

oil_raster = psm.rasters.df2raster(
    oil,
    "lon",
    "lat",
    "count",
    xyscale=xyscale,
    per_km2=False,
    origin="lower",
)

wind_raster = psm.rasters.df2raster(
    wind,
    "lon",
    "lat",
    "count",
    xyscale=xyscale,
    per_km2=False,
    origin="lower",
)

other_raster = psm.rasters.df2raster(
    other,
    "lon",
    "lat",
    "count",
    xyscale=xyscale,
    per_km2=False,
    origin="lower",
)

mara_raster = psm.rasters.df2raster(
    mara,
    "lon",
    "lat",
    "count",
    xyscale=xyscale,
    per_km2=False,
    origin="lower",
)


if 1:
    # For 1 deg resolution
    x = np.arange(-180, 180, 1) + 0.5
    y = np.arange(-90, 90, 1) + 0.5
else:
    # For 2 deg resolution
    x = np.arange(-180, 180, 2) + 1
    y = np.arange(-90, 90, 2) + 1

    oil_raster = 4 * skimage.transform.downscale_local_mean(oil_raster, (2, 2))
    wind_raster = 4 * skimage.transform.downscale_local_mean(
        wind_raster, (2, 2)
    )
    other_raster = 4 * skimage.transform.downscale_local_mean(
        other_raster, (2, 2)
    )

    print(wind_raster.shape)

# %%
# Convert rasters to vectors

xx, yy = np.meshgrid(x, y)
x_ = xx.flatten()
y_ = yy.flatten()

z_ = oil_raster.flatten()
(i,) = np.where(z_ > 0)
x_oil = x_[i]
y_oil = y_[i]
z_oil = z_[i]
order = np.argsort(-z_oil)  # sort in desc order
x_oil = np.take(x_oil, order)
y_oil = np.take(y_oil, order)
z_oil = np.take(z_oil, order)

z_ = wind_raster.flatten()
(i,) = np.where(z_ > 0)
x_wind = x_[i]
y_wind = y_[i]
z_wind = z_[i]
order = np.argsort(-z_wind)  # sort in desc order
x_wind = np.take(x_wind, order)
y_wind = np.take(y_wind, order)
z_wind = np.take(z_wind, order)

z_ = other_raster.flatten()
(i,) = np.where(z_ > 0)
x_other = x_[i]
y_other = y_[i]
z_other = z_[i]

z_ = mara_raster.flatten()
(i,) = np.where(z_ > 0)
x_mara = x_[i]
y_mara = y_[i]
z_mara = z_[i]


print(f"oil:   {len(z_oil)} {z_oil.min():.0f} {z_oil.max():.0f}")
print(f"wind:  {len(z_wind)} {z_wind.min():.0f} {z_wind.max():.0f}")
print(f"other: {len(z_other)} {z_other.min():.0f} {z_other.max():.0f}")
print(f"mara:  {len(z_mara)} {z_mara.min():.0f} {z_mara.max():.0f}")

# %%
# Load infrastructure time series
QUERY = True

tab_glob = "infra_global_time_series_v20230222"
tab_eez = "infra_eez_time_series_v20230222"

if QUERY:
    qg = f"SELECT * FROM proj_global_sar.{tab_glob}"
    qe = f"SELECT * FROM proj_global_sar.{tab_eez}"
    df_glob = pd.read_gbq(qg, project_id="world-fishing-827")
    df_eez = pd.read_gbq(qe, project_id="world-fishing-827") 
    df_glob.to_feather(f"../../data/{tab_glob}.feather")
    df_eez.to_feather(f"../../data/{tab_eez}.feather")
else:
    df_glob = pd.read_feather(f"../../data/{tab_glob}.feather", use_threads=True)
    df_eez = pd.read_feather(f"../../data/{tab_eez}.feather", use_threads=True)

df3 = df_glob
df4 = df_eez 

df_eez.head()

# %%
df3 = df3.ffill().bfill()
df4 = df4.ffill().bfill()
df4["country"] = df4.iso3.apply(get_country)
df4["continent"] = df4.iso3.apply(get_continent)

# %%
df3.head()


# %%
def get_date_years(df, time_col=None):
    
    def date_to_year(d):
        start = dt.date(d.year, 1, 1).toordinal()
        year_length = dt.date(d.year + 1, 1, 1).toordinal() - start
        return d.year + float(d.toordinal() - start) / year_length

    def get_frac_year(df, time_col=time_col):
        if time_col:
            return [date_to_year(d) for d in df[time_col]]
        else:
            return [date_to_year(d) for d in df.index.to_pydatetime()]

    df["date_year"] = get_frac_year(df)

    return df


for d in [df3, df4]:
    d = get_date_years(d, "detect_date")

# %%
# Make WIND time series

countries = [
    "CHN",
    "GBR",
    "DEU",
    "DNK",
    "NLD",
    # 'BEL',
    # 'SWE',
    # 'TWN',
]

dd = {}
for country in countries:

    d = df4[
        (df4.region == "inside_all_oil_polygons")
        | (df4.region == "outside_all_oil_polygons")
    ]

    dd[country] = (
        d[d.iso3 == country]
        .groupby(["date_year"])
        .sum()
        .rolling(3, center=True)
        .median()
    )

list(dd.values())[0].head()

# %%
SAVE = True
PLOT1 = True
PLOT2 = True
PLOT3 = True

plt.rcdefaults()
fig = plt.figure(figsize=(14, 6.5 * 3), constrained_layout=True)

grid1 = GridSpec(
    2,
    2,
    height_ratios=[1.1, 0.9],
    top=1,
    bottom=0.35,
    wspace=0.01,
    hspace=0.015,
    left=0.01,
    right=0.98,
)
grid2 = GridSpec(
    1,
    2,
    height_ratios=[0.9],
    top=0.30,
    bottom=0.05,
    wspace=0.12,
    hspace=0.2,
    left=0.07,
    right=0.96,
)

sub1 = grid1[(0, slice(0, 2))]
sub2 = grid1[(1, 0)]
sub3 = grid1[(1, 1)]
sub4 = grid2[(0, 0)]
sub5 = grid2[(0, 1)]

with psm.context(psm.styles.light):

    plt.rc("font", size=14)  # controls default text sizes
    plt.rc("axes", titlesize=14)  # fontsize of the axes title
    plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=14)  # fontsize of the tick labels
    plt.rc("legend", fontsize=14)  # legend fontsize

    # ===== Global Map ===== #

    if PLOT1:

        scale = 2.75
        alpha = 0.3

        prj = cartopy.crs.Robinson(central_longitude=0, globe=None)
        extent = (-140, 170, -60, 90)  # trucate global map

        ax1 = psm.create_map(subplot=sub1, projection=prj)
        ax1.set_extent(extent, crs=psm.identity)
        ax1.axis("off")

        psm.add_land(ax1, color="0.88")
        psm.add_countries(ax1, linewidth=0.5, facecolor=(0, 0, 0, 0))
        # psm.add_eezs(ax1)

        ax1.scatter(
            x=x_oil,
            y=y_oil,
            s=z_oil * scale,
            c="#2954E1",
            alpha=alpha,
            transform=psm.identity,
            zorder=2000,
        )
        ax1.scatter(
            x=x_wind,
            y=y_wind,
            s=z_wind * scale,
            c="#F77539",
            alpha=alpha,
            transform=psm.identity,
            zorder=1000,
        )
        ax1.scatter(
            x=x_other,
            y=y_other,
            s=z_other * scale,
            facecolors='#24BDBD',
            edgecolors='#24BDBD',
            transform=psm.identity,
            zorder=10,
        )
        ax1.scatter(
            x=x_mara,
            y=y_mara,
            s=z_mara * scale,
            facecolors='none',
            edgecolors="#2954E1",
            linewidth=0.75,
            transform=psm.identity,
            zorder=20,
        )

        # Here we create a legend:
        
        labels = [
            "Oil infrastructure",
            "Wind turbines",
            "Other structures",
        ]
        colors = ["#2954E1", "#F77539", "#24BDB2"]
        
        for label, color, ypos in zip(labels, colors, [0.35, 0.305, 0.26]):
            ax1.text(
                0.007,
                ypos,
                label,
                ha="left",
                color=color,
                fontsize=17,
                weight="normal",
                transform=ax1.transAxes,
            )
            
        # we'll plot empty lists with the desired size and label
        for area in [10, 100, 500]:
            plt.scatter(
                [],
                [],
                c="k",
                alpha=0.3,
                s=area * scale,
                label=f"{area}",
                transform=psm.identity,
            )
            
        plt.legend(
            scatterpoints=1,
            frameon=False,
            labelspacing=1,
            title="per 100 km$^2$",
            loc="lower left",
            borderaxespad=0.5,
        )

    # ===== Regional Maps ===== #

    if PLOT2:

        count = 0

        for sub, raster in zip([sub2, sub3], [raster_oil, raster_wind]):

            xc, yc, dx_x_2 = (5.0, 56.3, 10.5)  # x, y, zoom
            dx = dx_x_2 / 2
            extent = (xc - dx, xc + dx, yc - dx, yc + dx)

            prj = cartopy.crs.LambertAzimuthalEqualArea(xc, yc)
            ax = psm.create_map(subplot=sub, projection=prj)

            ax.set_extent(extent, crs=psm.identity)
            ax.set_adjustable("datalim")
            psm.add_land(ax, edgecolor='darkgray', facecolor='0.88', linewidth=0.5)
            psm.add_countries(ax, linewidth=1, facecolor=(0, 0, 0, 0))

            norm = mpcolors.LogNorm(vmin=1e-2, vmax=1e2)

            im = psm.add_raster(
                raster,
                ax=ax,
                cmap="YlGnBu",
                norm=norm,
                origin="lower",
            )

            if count == 0:
                ax.text(
                    0.97,
                    -0.066,
                    "Hours of vessel activity per km$^2$",
                    ha="right",
                    color="0.2",
                    fontsize=16,
                    transform=ax.transAxes,
                )
            elif count == 1:
                cb = psm.add_colorbar(
                    im,
                    ax=ax,
                    label="",
                    loc="bottom",
                    width=1,
                    height=0.025,
                    hspace=0.03,
                    wspace=0.016,
                    valign=0.5,
                    right_edge=None,
                    center=False,
                    # formatter="%.2f",
                )
                
            s = "Oil" if count == 0 else "Wind"
            ax.text(
                0.92,
                0.95,
                f"{s} vessel traffic",
                ha="right",
                va="top",
                color='0.2',
                weight="normal",
                fontsize=19,
                transform=ax.transAxes,
            )
            count += 1

    # ===== Time Series ===== #

    if PLOT3:

        color_wind = palette2[0]
        color_oil = "#4BA6AF"

        ax4 = fig.add_subplot(sub4)

        d_in = df3[df3.region == "inside_all_oil_polygons"]
        d_out = df3[df3.region == "outside_all_oil_polygons"]

        d_in = d_in.rolling(3, center=True).median()
        d_out = d_out.rolling(3, center=True).median()

        d_in = d_in.iloc[2:, :]
        d_out = d_out.iloc[2:, :]

        y1 = d_in.oil + d_in.other.values
        y1_max = d_in.oil + d_in.other.values + d_out.oil.values
        y1_min = d_in.oil
        y2 = d_in.wind + d_out.wind.values
        x = d_in.date_year

        y1_err = (y1_max - y1) / 2.0
        y1_mid = y1 + y1_err

        ax4.errorbar(
            x.values[::2],
            y1_mid.values[::2],
            yerr=y1_err[::2],
            color=color_oil,
            ls="none",
            linewidth=2.5,
            capsize=6,
            capthick=2,
        )
        ax4.plot(x, y2, color=color_wind, linewidth=2.5)  # WIND
        ax4.set_ylim(4400, 11000)

        # ax4.set_ylabel("Number of detected structures in the ocean")
        ax4.text(
            -0.125,
            0.5,
            "Number of detected structures in the ocean",
            ha="right",
            va="center",
            color=color_wind,
            weight="normal",
            fontsize=15,
            rotation=90,
            transform=ax4.transAxes,
        )
        ax4.text(
            0.04,
            0.53,
            "Oil infrastructure",
            ha="left",
            color=color_oil,
            fontsize=16,
            transform=ax4.transAxes,
        )
        ax4.text(
            0.20,
            0.05,
            "Wind turbines",
            ha="left",
            color=color_wind,
            fontsize=16,
            transform=ax4.transAxes,
        )

        #####

        ax5 = fig.add_subplot(sub5)

        indices = np.linspace(0.2, 1.2, len(countries))  # for colors

        for k, (n, d, i) in enumerate(zip(dd.keys(), dd.values(), indices)):
            ax5.plot(
                d.index,
                d.wind,
                label=n,
                color=palette4[::-1][k],
                linewidth=2.5,
            )
            ax5.set_ylim(100, 5000)

        legend = ax5.legend(frameon=False)
        for k, text in enumerate(legend.get_texts()):
            text.set_color(palette4[::-1][k])

        #####

        # Hide the right and top spines
        for ax in [ax4, ax5]:
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.minorticks_off()
            # ax.grid(axis="y")

        ax5.text(
            0.04,
            0.45,
            "Wind turbines",
            ha="left",
            color=color_wind,
            fontsize=16,
            transform=ax5.transAxes,
        )

if SAVE:
    plt.savefig(
        "../../figures/fig4v3.png", bbox_inches="tight", pad_inches=0.05, dpi=300
    )

# %%
