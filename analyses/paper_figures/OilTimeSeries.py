# %% [markdown]
# ## V8 (using interpolated data, and revised infrastructure)
# 
# Time series of oil 

# %%
import datetime
import warnings
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot as pplt
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
import statsmodels.api as sm
from scipy import stats
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
%matplotlib inline

# %%
import pycountry
from pycountry_convert import (
    country_alpha2_to_continent_code,
    country_name_to_country_alpha2,
)

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


# https://www.learnui.design/tools/data-color-picker.html
palette1 = [
    "#003f5c",
    "#58508d",
    "#bc5090",
    "#ff6361",
    "#ffa600",
]
palette2 = [
    # '#003f5c',
    "#444e86",
    "#955196",
    "#dd5182",
    "#ff6e54",
    "#ffa600",
]
palette3 = [
    "#004c6d",
    "#346888",
    "#5886a5",
    "#7aa6c2",
    "#9dc6e0",
    # '#c1e7ff',
]
palette4 = [
    "#90e28d",
    "#2cbd9b",
    "#00949b",
    "#006a87",
    "#1f4260",
]
palette5 = [
    "#374c80",
    "#7a5195",
    "#bc5090",
    "#ef5675",
    "#ff764a",
]

mycmap1 = get_continuous_cmap(palette1)
mycmap2 = get_continuous_cmap(palette2)
mycmap3 = get_continuous_cmap(palette3)
mycmap4 = get_continuous_cmap(palette4)
mycmap5 = get_continuous_cmap(palette5)

# %%
# NOTE: feather can store date objects, CSV dates need to be parsed (str -> date)
df2 = pd.read_csv(
    "../data/infra_global_time_series_v20230214.csv", parse_dates=["detect_date"]
)
df3 = pd.read_csv(
    "../data/infra_poly_time_series_v20230214.csv", parse_dates=["detect_date"]
)
df4 = pd.read_csv(
    "../data/infra_eez_time_series_v20230214.csv", parse_dates=["detect_date"]
)

# %%
df2 = df2.ffill().bfill()
df3 = df3.ffill().bfill()
df4 = df4.ffill().bfill()

df4["country"] = df4.iso3.apply(get_country)
df4["continent"] = df4.iso3.apply(get_continent)

# %%
# Get year intervals, and fractional year time index


def get_date_years(df, time_col=None):
    def date_to_year(d):
        start = datetime.date(d.year, 1, 1).toordinal()
        year_length = datetime.date(d.year + 1, 1, 1).toordinal() - start
        return d.year + float(d.toordinal() - start) / year_length

    def get_frac_year(df, time_col=time_col):
        if time_col:
            return [date_to_year(d) for d in df[time_col]]
        else:
            return [date_to_year(d) for d in df.index.to_pydatetime()]

    df["date_year"] = get_frac_year(df)

    return df


for d in [df2, df3, df4]:
    d = get_date_years(d, "detect_date")

# %% [markdown]
# ## WIND

# %%
# Select wind countries to plot

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
    inside = "inside_all_oil_polygons"
    outside = "outside_all_oil_polygons"
    d = df4[(df4.region == inside) | (df4.region == outside)]

    dd[country] = (
        d[d.iso3 == country]
        .groupby(["date_year"])
        .sum()
        .rolling(3, center=True)
        .median()
    )

list(dd.values())[0].head()

# %% [markdown]
# ## OIL

# %%
# Make annual time series per country

countries1 = [
    "USA",
    "SAU",
    "IDN",
    "ARE",
    "THA",
    "NGA",
    "CHN",
    "MYS",
    'AGO',
    "TTO",
]
countries2 = [
    'MEX',
    'IND',
    'GBR',
    'QAT',
    'IRN',
    'EGY',
    'BRA',
    'BRN',
    'GAB',
    'NLD',
]

# One df per country
def get_series(df4, countries):
    dd = {}
    for country in countries:
        # print(country)
        d = df4[df4.region == "inside_all_oil_polygons"]
        dd[country] = (
            d[d.iso3 == country]
            .groupby(["date_year"])
            .sum()
        )
    return dd
        
    
dd1 = get_series(df4, countries1)
dd2 = get_series(df4, countries2)

list(dd1.values())[0].head()

# %%
def get_trend(x):
    """Piecewise polynomila fit.
    
    For quarterly data Hodrick and Prescott [1] used Smoothing=1600.
    For monthly data a commonly accepted value in the literature is 14400.
    For daily data the Smoothing parameter can be anywhere from 10000 to 10^8.
    """
    return sm.tsa.filters.hpfilter(x, 129600)  # 14400)


def get_outliers(t, x, nstd=1, plot=False):
    cycle, trend = get_trend(x)
    resid = x - trend
    if plot:
        plt.figure()
        plt.plot(t, x, t, trend)
        plt.show()
    return np.where(np.abs(resid) > nstd * np.nanstd(resid))[0]


def filt_spikes(dd, nstd=3, niter=2, interp=True):
    """Fit trend and remove residuals > nstd [iteratively]"""
    dm = {}
    for (k, v) in zip(dd.keys(), dd.values()):
        v = v.astype(float)
        for _ in range(niter):
            i_out = get_outliers(v.index.values, v.oil.values, nstd=nstd)
            v.iloc[i_out] = np.nan
            if interp:
                v = v.interpolate().bfill()
        dm[k] = v
    return dm


def get_medians(dd, quarterly=False):
    dm = {}
    for (k, v) in zip(dd.keys(), dd.values()):
        years = [2017, 2018, 2019, 2020, 2021, 2022]
        if quarterly:
            ys = years.copy()
            [ys.insert(i*2 + 1, years[i]+0.5) for i in range(len(years))]
            years = ys
        dt = years[1] - years[0]
        for year in years:
            cond = (v.index >= year - dt) & (v.index < year)
            v.oil[cond] = v.oil[cond].median()
        dm[k] = v.ffill().bfill()
        print(f"{k}  {np.nanmedian(v):.0f}  {np.nanmean(v):.0f}")
    return dm


def get_rolling(dd, window, op="median", center=True):
    dm = {}
    for (k, v) in zip(dd.keys(), dd.values()):
        v = v.astype(float)
        if op == 'median':
            v = v.rolling(window, center=center).median()
        if op == 'mean':
            v = v.rolling(window, center=center).mean()
        dm[k] = v.bfill().ffill()
    return dm


def plot_series(dd, ax=None):
    mycmap4_r = mycmap4.reversed()
    for i, (n, d) in enumerate(zip(dd.keys(), dd.values())):
        print(i, n)
        ax.plot(
            d.index,
            d.oil,
            label=n,
            # color=mycmap1(i/N1),
            color=mycmap4_r(i/N1),
            linewidth=2.75,
        )
    return ax


SAVE = False
N1 = len(countries1)
N2 = len(countries2)

dd1 = filt_spikes(dd1, nstd=2, niter=2, interp=True)
dd2 = filt_spikes(dd2, nstd=2, niter=2, interp=True)

# Comment if plotting raw series
dd1 = get_medians(dd1, quarterly=False)
dd2 = get_medians(dd2, quarterly=False)

# dd1 = get_rolling(dd1, 5, op='median')
# dd2 = get_rolling(dd2, 5, op='median')


# ======= PLOT ======= #

plt.figure(figsize=(10, 10))

ax1 = plt.subplot(121)
plot_series(dd1, ax1)
plt.legend(bbox_to_anchor=(0, 1), loc="upper left", ncol=1, frameon=False)
plt.ylabel("Number of detected oil structures", fontsize=12)
# plt.grid(False)

ax2 = plt.subplot(122)
plot_series(dd2, ax2)
plt.legend(bbox_to_anchor=(0, 1), loc="upper left", ncol=1, frameon=False)
# plt.grid(False)
plt.ylim(100, 220)


for ax in [ax1, ax2]:
    # Move left and bottom spines outward by 10 points
    ax.spines.left.set_position(('outward', 10))
    ax.spines.bottom.set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


if SAVE:
    plt.savefig("../figures/oilseries4.png", bbox_inches="tight", pad_inches=0.25, dpi=300)

# %%
# It figures out the correct height ratios of (broken) subplots

SAVE = True

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(7.75, 8.2))

ylim1 = (2110, 2430)  # outliers only
ylim2 = (200, 780)  # most of the data
ylim3 = (100, 220)

ylim1ratio = (ylim1[1]-ylim1[0])/(ylim2[1]-ylim2[0]+ylim1[1]-ylim1[0])
ylim2ratio = (ylim2[1]-ylim2[0])/(ylim2[1]-ylim2[0]+ylim1[1]-ylim1[0])
gs = gridspec.GridSpec(2, 2, height_ratios=[ylim1ratio, ylim2ratio])

# ======= left plots ======= #

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[2])

ax1 = plot_series(dd1, ax1)
ax2 = plot_series(dd1, ax2)

ax1.set_ylim(*ylim1)  # outliers only
ax2.set_ylim(*ylim2)  # most of the data

ax1.spines[['right', 'top', 'bottom']].set_visible(False)
ax2.spines[['right', 'top']].set_visible(False)
ax1.xaxis.set_ticks([])
ax2.yaxis.set_ticks(range(*ylim2, 50))
ax1.minorticks_off()
ax2.minorticks_off()

# Single y-axis label
plt.subplots_adjust(hspace=0.03)
ax2.set_ylabel('Number of detected oil structures', size=12)
ax2.yaxis.set_label_coords(0.05, 0.5, transform=fig.transFigure)

# Mark the y-axis discontinuity
kwargs = dict(color='k', clip_on=False)
xlim = ax1.get_xlim()
dx = .02*(xlim[1]-xlim[0])
dy = .01*(ylim1[1]-ylim1[0])/ylim1ratio
ax1.plot((xlim[0]-dx,xlim[0]+dx), (ylim1[0]-dy,ylim1[0]+dy), **kwargs)
# ax1.plot((xlim[1]-dx,xlim[1]+dx), (ylim1[0]-dy,ylim1[0]+dy), **kwargs)
dy = .01*(ylim2[1]-ylim2[0])/ylim2ratio
ax2.plot((xlim[0]-dx,xlim[0]+dx), (ylim2[1]-dy,ylim2[1]+dy), **kwargs)
# ax2.plot((xlim[1]-dx,xlim[1]+dx), (ylim2[1]-dy,ylim2[1]+dy), **kwargs)
ax1.set_xlim(xlim)
ax2.set_xlim(xlim)

ax1.legend(loc="upper right", frameon=False, labelcolor='linecolor')
ax1.grid(axis='x')
ax2.grid(axis='x')

# ======= rigth plot ======= #

ax3 = fig.add_subplot(gs[:, 1])
ax3 = plot_series(dd2, ax3)
ax3.set_ylim(*ylim3)  # most of the data

ax3.spines[['right', 'top']].set_visible(False)
ax3.yaxis.set_ticks(range(*ylim3, 10))
ax3.minorticks_off()

ax3.legend(loc="upper left", frameon=False, labelcolor='linecolor')
ax3.grid(axis='x')

    
plt.tight_layout()

if SAVE:
    plt.savefig("figures/oilseries6b.png", bbox_inches="tight", pad_inches=0.1, dpi=300)

# %%



