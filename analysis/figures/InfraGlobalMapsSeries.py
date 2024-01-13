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
# import cartopy
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import datetime as dt
import pyseas.maps as psm
import skimage
import cartopy
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec

# %matplotlib inline

# %%
# Local imports
from cmaps import *

palette4 = [
    # "#fc8d59",
    # "#ef6548",
    # "#d7301f",
    # "#b30000",
    # "#7f0000",
    #
"#fd8d3c",
"#fc4e2a",
"#e31a1c",
"#bd0026",
"#800026",
]

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
import pandas as pd
from io import BytesIO
from zipfile import ZipFile

def read_zip_to_dataframes(zip_file_path):
    with ZipFile(zip_file_path, 'r') as zip_file:
        for file_name in zip_file.namelist():
            if file_name.endswith('.csv'):
                with zip_file.open(file_name) as csv_file:
                    csv_content = BytesIO(csv_file.read())
                    df = pd.read_csv(csv_content)
                    yield df



# %%
# Load data on vessel traffic

data_oil = '../data/infra_vessel_activity_oil_100th_degree_v20230222.zip'
data_wind = '../data/infra_vessel_activity_wind_100th_degree_v20230222.csv'


df_oil = pd.concat(read_zip_to_dataframes(data_oil), ignore_index=True)
df_wind = pd.read_csv(data_wind)

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
            f"{round(d.visit_hours_sum.min(), 4): >10}",
            f"{round(d.visit_hours_sum.max(), 1): >10}",
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
# Load data for infrastructure map

data_infra = "../data/offshore_infra_reclassified_w_regions_v20230816.csv"

df2 = pd.read_csv(data_infra)

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
print('oil:', len(oil))
print('wind:', len(wind))
print('other:', len(other))
print('mara:', len(mara))

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

data_glob = "../data/infra_global_time_series_v20230816.csv"
data_eez = "../data/infra_eez_time_series_v20230816.csv"

df3 = pd.read_csv(data_glob, parse_dates=['detect_date'])
df4 = pd.read_csv(data_eez, parse_dates=['detect_date'])

# %%
df4["country"] = df4.iso3.apply(get_country)
df4["continent"] = df4.iso3.apply(get_continent)

# %%
df3.head()

# %%
df4.head()


# %%
# WARNING: if AttributeError: 'str' object has no attribute 'year'
# parse CSV date column as datetime object

def get_date_years(df, time_col=None):
    
    def date_to_year(dd):
        start = dt.date(dd.year, 1, 1).toordinal()
        year_length = dt.date(dd.year + 1, 1, 1).toordinal() - start
        return dd.year + float(dd.toordinal() - start) / year_length

    def get_frac_year(df, time_col=time_col):
        if time_col:
            return [date_to_year(d) for d in df[time_col]]
        else:
            return [date_to_year(d) for d in df.index.to_pydatetime()]

    df["date_year"] = get_frac_year(df)

    return df


df3 = get_date_years(df3, "detect_date")
df4 = get_date_years(df4, "detect_date")


# %%
# Make WIND time series

def get_trend(x):
    """Piecewise polynomila fit.
    
    For quarterly data Hodrick and Prescott [1] used Smoothing=1600.
    For monthly data a commonly accepted value in the literature is 14400.
    For daily data the Smoothing parameter can be anywhere from 10000 to 10^8.
    """
    return sm.tsa.filters.hpfilter(x, 129600)


def get_outliers(t, x, nstd=1, plot=False):
    cycle, trend = get_trend(x)
    resid = x - trend
    if plot:
        plt.figure()
        plt.plot(t, x, t, trend)
        plt.show()
    return np.where(np.abs(resid) > nstd * np.nanstd(resid))[0]


def filt_spikes(dd, nstd=3, niter=2, interp=True, key=''):
    """Fit trend and remove residuals > nstd [iteratively]"""
    dm = {}
    for (k, v) in zip(dd.keys(), dd.values()):
        
        if key and k != key:
            dm[k] = v
            continue
            
        v = v.astype(float)
        for _ in range(niter):
            i_out = get_outliers(v.index.values, v.oil.values, nstd=nstd)
            v.iloc[i_out] = np.nan
            if interp:
                v = v.interpolate().bfill()
        dm[k] = v
    return dm


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

# One df per country
# dfs is only for wind plots
dfs = {}  

for country in countries:
    dfs[country] = (
        df4[df4.iso3 == country]
        .groupby(["date_year"])
        .sum()
    )
    
# Filter time series
# dfs = filt_spikes(dfs, nstd=1, niter=2, interp=True, key='GBR')
 
for k, v in dfs.items():
    if k == 'GBR':
        v = v.rolling(3, center=True).mean()
        dfs[k] = v

list(dfs.values())[0].head()


# %%
def add_names(names, ax):
    for (x,y,z,c) in names:
        ax.text(
            x,
            y,
            z,
            ha="right",
            color=c,
            fontsize=FONT,
            transform=ax.transAxes,
            path_effects=[pe.withStroke(linewidth=2, foreground="w")],
        )
        
        
def mouse_event(event):
    print(f"{event.xdata:.0f}, {event.ydata:.0f}")


# %%
# %matplotlib inline

SAVE = True
PLOT1 = True
PLOT2 = True
PLOT3 = True
FONT = 9

# colors = ["#2954E1", "#F77539", "#24BDB2"]
c_oil = "#7570b3"
c_wind = "#d95f02"
c_other = "#1b9e77"

# c_oil = "#8da0cb"
# c_wind = "#fc8d62"
# c_other = "#66c2a5"

scl = 0.5

plt.rcdefaults()
fig = plt.figure(figsize=(14 * scl, 6.5 * 3 * scl), constrained_layout=True)
cid = fig.canvas.mpl_connect('button_press_event', mouse_event)

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
    left=0.09,
    right=0.98,
)

sub1 = grid1[(0, slice(0, 2))]
sub2 = grid1[(1, 0)]
sub3 = grid1[(1, 1)]
sub4 = grid2[(0, 0)]
sub5 = grid2[(0, 1)]

with psm.context(psm.styles.light):

    plt.rc("font", size=FONT)  # controls default text sizes
    plt.rc("axes", titlesize=FONT)  # fontsize of the axes title
    plt.rc("axes", labelsize=FONT)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=FONT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=FONT)  # fontsize of the tick labels
    plt.rc("legend", fontsize=FONT)  # legend fontsize

    # ===== Global Map ===== #

    if PLOT1:

        scale = 2.5 * scl
        alpha = 0.4

        prj = cartopy.crs.Robinson(central_longitude=0, globe=None)
        extent = (-140, 170, -60, 90)  # trucate global map

        ax1 = psm.create_map(subplot=sub1, projection=prj)
        ax1.set_extent(extent, crs=psm.identity)
        ax1.axis("off")

        psm.add_land(ax1, color="0.88")
        psm.add_countries(ax1, linewidth=0.5 * scl, facecolor=(0, 0, 0, 0))
        # psm.add_eezs(ax1)

        ax1.scatter(
            x=x_oil,
            y=y_oil,
            s=z_oil * scale,
            c=c_oil, # "#2954E1",
            alpha=alpha,
            transform=psm.identity,
            zorder=2000,
        )
        ax1.scatter(
            x=x_wind,
            y=y_wind,
            s=z_wind * scale,
            c=c_wind, # "#F77539",
            alpha=alpha,
            transform=psm.identity,
            zorder=1000,
        )
        ax1.scatter(
            x=x_other,
            y=y_other,
            s=z_other * scale,
            facecolors=c_other, # '#24BDBD',
            edgecolors=c_other, # '#24BDBD',
            transform=psm.identity,
            zorder=10,
        )
        ax1.scatter(
            x=x_mara,
            y=y_mara,
            s=z_mara * scale,
            c=c_oil, # "#2954E1",
            alpha=alpha - 0.05,
            transform=psm.identity,
            zorder=2,
        )

        # Here we create a legend:
        
        labels = [
            "Oil infrastructure",
            "Wind turbines",
            "Other structures",
        ]
        colors = [c_oil, c_wind, c_other]
        
        for label, color, ypos in zip(labels, colors, [0.45, 0.40, 0.35]):
            ax1.text(
                0.007,
                ypos,
                label,
                ha="left",
                color=color,
                fontsize=FONT,
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
        
        ax1.text(0.01, 0.98, 'a', fontsize=FONT+1, weight='bold', ha='left', va='top', transform=ax1.transAxes)

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
            psm.add_land(ax, edgecolor='darkgray', facecolor='0.88', linewidth=0.5 * scl)
            psm.add_countries(ax, linewidth=1 * scl, facecolor=(0, 0, 0, 0))

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
                    color="0.1",
                    fontsize=FONT,
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
#                     center=False,
                    # formatter="%.2f",
                )
                
            s = "Oil" if count == 0 else "Wind"
            ax.text(
                0.92,
                0.95,
                f"{s} vessel traffic",
                ha="right",
                va="top",
                color='0.1',
                weight="normal",
                fontsize=FONT,
                transform=ax.transAxes,
            )
            
            x, y, s = (177472, 412876, 'NO')
            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
            x, y, s = (535756, 252266, 'SE')
            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
            x, y, s = (259836, -19536, 'DK')
            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
            x, y, s = (387500, -431356,'DE')
            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
            x, y, s = (41571, -443711, 'NL')
            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
            x, y, s = (-452613, -365465, 'GB')
            ax.text(x, y, s, ha='center', va='center', color='0.6', rotation=0, style='normal', fontsize=FONT)
            
            if count == 0:
                ax.text(0.02, 0.98, 'b', fontsize=FONT+1, weight='bold', ha='left', va='top', transform=ax.transAxes)
        
            count += 1

    # ===== Time Series ===== #

    if PLOT3:

        ax4 = fig.add_subplot(sub4)

        # Filter time series for spikes
        df3_filt = df3.rolling(3, center=True).median()
        
        y1_min = df3_filt.oil
        y1_max = df3_filt.oil + df3_filt.probable_oil.values + df3_filt.possible_oil.values
        
        y1_err = (y1_max - y1_min) / 2
        y1_mid = y1_min + y1_err

        y2 = df3_filt.wind
        x = df3_filt.date_year

        ax4.errorbar(
            x.values[::2],  # plot every other error bar
            y1_mid.values[::2],
            yerr=y1_err[::2],
            color=c_oil,
            ls="none",
            linewidth=2.6 * scl,
            capsize=6 * scl,
            capthick=2 * scl,
        )
        ax4.plot(x, y2, color=c_wind, linewidth=2.6 * scl)  # WIND
        ax4.set_ylim(4400, 11000)

        ax4.text(
            -0.15,
            0.5,
            "Number of detected structures in the ocean",
            ha="right",
            va="center",
            color='0.1',
            weight="normal",
            fontsize=FONT,
            rotation=90,
            transform=ax4.transAxes,
        )
        ax4.text(
            0.04,
            0.45,
            "Oil infrastructure",
            ha="left",
            color=c_oil,
            fontsize=FONT,
            transform=ax4.transAxes,
        )
        ax4.text(
            0.20,
            0.05,
            "Wind turbines",
            ha="left",
            color=c_wind,
            fontsize=FONT,
            transform=ax4.transAxes,
        )
        
        
        ax4.text(-0.17, 1.14, 'c', fontsize=FONT+1, weight='bold', ha='left', va='top', transform=ax4.transAxes)

        #####

        ax5 = fig.add_subplot(sub5)

        indices = np.linspace(0.2, 1.2, len(countries))  # for colors

        for k, (n, d, i) in enumerate(zip(dfs.keys(), dfs.values(), indices)):
            ax5.plot(
                d.index,
                d.wind,
                label=n,
                color=palette4[::-1][k],
                linewidth=2.6 * scl,
            )
            ax5.set_ylim(100, 5000)

        c = [palette4[::-1][k] for k in range(5)]
        names = [
            (0.96, 0.98, 'China', c[0]),
            (0.96, 0.55, 'United Kingdom', c[1]),
            (0.96, 0.32, 'Germany', c[2]),
            (0.96, 0.13, 'Denmark', c[3]),
            (0.96, 0.015, 'Netherlands', c[4]),
        ]
        add_names(names, ax5)

        #####

        # Hide the right and top spines
        for ax in [ax4, ax5]:
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.minorticks_off()
            # ax.grid(axis="y")

        ax5.text(
            0.05,
            0.95,
            "Wind turbines\ntop nations",
            ha="left",
            va="top",
            color=c_wind,
            fontsize=FONT,
            transform=ax5.transAxes,
        )

if SAVE:
    plt.savefig(
        "./infra_global_maps_series_v3.jpg",
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=300
    )

# %%
