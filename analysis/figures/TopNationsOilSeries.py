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

from cmaps import * 

warnings.filterwarnings("ignore")
%matplotlib inline

# %%
# palette = [
#     "#a6cee3",
#     "#1f78b4",
#     "#b2df8a",
#     "#33a02c",
#     "#fb9a99",
#     "#e31a1c",
#     "#fdbf6f",
#     "#ff7f00",
#     "#cab2d6",
#     "#6a3d9a",
# ]
# 
# mycmap = get_continuous_cmap(palette, n=10)

mycmap = mpl.cm.tab20c_r

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
# NOTE: feather can store date objects, CSV dates need to be parsed (str -> date)
f2 = "../data/infra_global_time_series_v20230816.csv"
# f3 = "../data/infra_poly_time_series_v20230816.csv"
f4 = "../data/infra_eez_time_series_v20230816.csv"

df2 = pd.read_csv(f2, parse_dates=["detect_date"])
# df3 = pd.read_csv(f3, parse_dates=["detect_date"])
df4 = pd.read_csv(f4, parse_dates=["detect_date"])

# %%
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


for d in [df2, df4]:
    d = get_date_years(d, "detect_date")

# %%
df2 = df2[df2.date_year < 2022]
# df3 = df3[df3.date_year < 2022]
df4 = df4[df4.date_year < 2022]

# %%
# Make annual time series per country

countries1 = [
    "USA",
    "SAU",
    "IDN",
    "ARE",
    "NGA",
    "THA",
    "MYS",
    "TTO",
    "CHN",
    'AGO',
]
countries2 = [
    'MEX',
    'IND',
    'EGY',
    'QAT',
    'IRN',
    'GBR',
    'BRA',
    'BRN',
    'GAB',
    'NLD',
]

# Code -> Name
code2name = {
    "USA": 'United States',
    "SAU": 'Saudi Arabia',
    "IDN": 'Indonesia',
    "ARE": 'United Arab Emirates',
    "NGA": 'Nigeria',
    "THA": 'Thailand',
    "MYS": 'Malaysia',
    "TTO": 'Trinidad and Tobago',
    "CHN": 'China',
    'AGO': 'Angola',
    'MEX': 'Mexico',
    'IND': 'India',
    'EGY': 'Egypt',
    'QAT': 'Qatar',
    'IRN': 'Iran',
    'GBR': 'United Kindom',
    'BRA': 'Brazil',
    'BRN': 'Brunei Darussalam',
    'GAB': 'Gabon',
    'NLD': 'Netherlands',
}

N1 = len(countries1)
N2 = len(countries2)

# One df per country
def get_series(df4, countries):
    dfs = {}
    for country in countries:
        dfs[country] = (
            df4[df4.iso3 == country]
            .groupby(["date_year"])
            .sum()
        )
    return dfs
        
    
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
    mycmap_r = mycmap.reversed()
    for i, (n, d) in enumerate(zip(dd.keys(), dd.values())):
        n = code2name[n]
        print(i, n)
        ax.plot(
            d.index,
            d.oil,
            label=n,
            color=mycmap_r(i/N1),
            linewidth=2.75,
        )
    return ax


# %%
# Filter time series

dd1 = filt_spikes(dd1, nstd=2, niter=2, interp=True)
dd2 = filt_spikes(dd2, nstd=2, niter=2, interp=True)

# Comment if plotting raw series
dd1 = get_medians(dd1, quarterly=False)
dd2 = get_medians(dd2, quarterly=False)

# dd1 = get_rolling(dd1, 5, op='median')
# dd2 = get_rolling(dd2, 5, op='median')

# %%
# NOTE: Plot all time series in absolute/continous y-axis

# SAVE = False
# 
# plt.figure(figsize=(10, 10))
# 
# ax1 = plt.subplot(121)
# plot_series(dd1, ax1)
# plt.legend(bbox_to_anchor=(0, 1), loc="upper left", ncol=1, frameon=False)
# plt.ylabel("Number of detected oil structures", fontsize=12)
# # plt.grid(False)
# 
# ax2 = plt.subplot(122)
# plot_series(dd2, ax2)
# plt.legend(bbox_to_anchor=(0, 1), loc="upper left", ncol=1, frameon=False)
# # plt.grid(False)
# plt.ylim(100, 220)
# 
# 
# for ax in [ax1, ax2]:
#     # Move left and bottom spines outward by 10 points
#     ax.spines.left.set_position(('outward', 10))
#     ax.spines.bottom.set_position(('outward', 10))
#     # Hide the right and top spines
#     ax.spines.right.set_visible(False)
#     ax.spines.top.set_visible(False)
#     # Only show ticks on the left and bottom spines
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')
# 
# 
# if SAVE:
#     plt.savefig(".././oilseries4.png", bbox_inches="tight", pad_inches=0.25, dpi=300)

# %%
# NOTE: It figures out the correct height ratios of (broken) subplots

SAVE = False

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(7.75, 8.2))

ylim1 = (2125, 2470)  # outliers only
ylim2 = (200, 780)  # most of the data
ylim3 = (110, 220)

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

ax1.legend(bbox_to_anchor=(1.03, 1.28), loc="upper right", frameon=False, labelcolor='linecolor', fontsize=10)
ax1.grid(axis='x')
ax2.grid(axis='x')

# ======= rigth plot ======= #

ax3 = fig.add_subplot(gs[:, 1])
ax3 = plot_series(dd2, ax3)
ax3.set_ylim(*ylim3)  # most of the data

ax3.spines[['right', 'top']].set_visible(False)
ax3.yaxis.set_ticks(range(*ylim3, 10))
ax3.minorticks_off()

ax3.legend(bbox_to_anchor=(0.05, 1.1), loc="upper left", frameon=False, labelcolor='linecolor', fontsize=10)
ax3.grid(axis='x')

    
plt.tight_layout()

if SAVE:
    plt.savefig(
        "./top_nations_oil_series_v3.jpg",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=300
    )

# %%
