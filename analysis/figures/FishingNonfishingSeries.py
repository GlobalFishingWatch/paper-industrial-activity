# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
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

import pycountry

from pycountry_convert import (
    country_alpha2_to_continent_code,
    country_name_to_country_alpha2,
    country_alpha2_to_country_name,
)

warnings.filterwarnings("ignore")
# %matplotlib inline

# +
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


# -

# NOTE: feather can store date objects, CSV dates need to be parsed (str -> date)
f = "../data/24day_rolling_augmented_v20230816.csv"
df1 = pd.read_csv(f, parse_dates=["rolling_date"])

# Crop beguining and end data points
cond1 = df1.rolling_date >= datetime.datetime(2017, 1, 13)
cond2 = df1.rolling_date < datetime.datetime(2021, 12, 20)
df1 = df1[cond1 & cond2]

print(df1.head())

dfW = df1[df1.eez_iso3 != "CHN"].groupby("rolling_date").sum()  # Ouside China
dfC = df1[df1.eez_iso3 == "CHN"].groupby("rolling_date").sum()  # Inside China
dfA = df1.groupby('rolling_date').sum()                         # All activity


# +
# Get year intervals, and fractional year time index


def get_year_intervals(df):
    i2017 = ["2017" for i in df.index.values.astype(str) if "2017" in i]
    i2018 = ["2018" for i in df.index.values.astype(str) if "2018" in i]
    i2019 = ["2019" for i in df.index.values.astype(str) if "2019" in i]
    i2020 = ["2020" for i in df.index.values.astype(str) if "2020" in i]
    i2021 = ["2021" for i in df.index.values.astype(str) if "2021" in i]
    i2022 = ["2022" for i in df.index.values.astype(str) if "2022" in i]
    df["year_interval"] = i2017 + i2018 + i2019 + i2020 + i2021 + i2022
    return df


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


for d in [dfW, dfC, dfA]:
    d = get_year_intervals(d)
    d = get_date_years(d)


# +
# Fit linear trend and cycle


def get_annual_means(df, calc_key):
    years = ["2017", "2018", "2019", "2020", "2021"]
    df[calc_key + "_mean"] = np.zeros(len(df))
    df[calc_key + "_std"] = np.zeros(len(df))
    for year in years:
        df[calc_key + "_mean"][df.year_interval == year] = df[calc_key][
            df.year_interval == str(year)
        ].mean()
        df[calc_key + "_std"][df.year_interval == year] = df[calc_key][
            df.year_interval == str(year)
        ].std()
    return df


def get_trend_and_cycle(df, calc_key):
    """
    A value of 1600 is suggested for quarterly data.
    Ravn and Uhlig suggest using a value of 6.25 (1600/4**4) for annual data
    and 129600 (1600*3**4) for monthly data.

    For quarterly data Hodrick and Prescott [1] used Smoothing=1600.
    For monthly data a commonly accepted value in the literature is 14400.
    For daily data the Smoothing parameter can be anywhere from 10000 to 10^8.
    """
    cycle, trend = sm.tsa.filters.hpfilter(df[calc_key], 10**12)
    df[calc_key + "_trend"] = trend
    df[calc_key + "_cycle"] = cycle
    return df


calc_keys = ["fishing", "nonfishing", "nonfishing100"]

for d in [dfW, dfC, dfA]:
    for calc_key in calc_keys:
        d = get_annual_means(d, calc_key)
        d = get_trend_and_cycle(d, calc_key)
# -

# Save time series data for uncertainty estimation 
# See BootstrapUncertaintySeries.py
if 0:
    cols = ['fishing', 'nonfishing']
    dfW[cols].to_csv('ts_world.csv', index=True)
    dfC[cols].to_csv('ts_china.csv', index=True)
    dfA[cols].to_csv('ts_global.csv', index=True)

# +
# Calculate statistics

# Use same year_interval for all
pre = (dfW.year_interval == "2018") | (dfW.year_interval == "2019")
pos = (dfW.year_interval == "2020") | (dfW.year_interval == "2021")
dt = (dfW.date_year.values[-1] - dfW.date_year.values[0])  

n_pre = fishW_pre = len(dfW.fishing[pre])
n_pos = fishW_pos = len(dfW.fishing[pos])

# Outside China
fishW_pre = dfW.fishing[pre].mean()
fishW_pos = dfW.fishing[pos].mean()
fishW_pre_sd = dfW.fishing[pre].std()
fishW_pos_sd = dfW.fishing[pos].std()
fishW_mean = dfW.fishing.mean()
fishW_trend = (dfW.fishing_trend.values[-1] - dfW.fishing_trend.values[0]) / dt

# Inside China
fishC_pre = dfC.fishing[pre].mean()
fishC_pos = dfC.fishing[pos].mean()
fishC_pre_sd = dfC.fishing[pre].std()
fishC_pos_sd = dfC.fishing[pos].std()
fishC_mean = dfC.fishing.mean()
fishC_trend = (dfC.fishing_trend.values[-1] - dfC.fishing_trend.values[0]) / dt

# All
fishA_pre = dfA.fishing[pre].mean()
fishA_pos = dfA.fishing[pos].mean()
fishA_pre_sd = dfA.fishing[pre].std()
fishA_pos_sd = dfA.fishing[pos].std()
fishA_mean = dfA.fishing.mean()
fishA_trend = (dfA.fishing_trend.values[-1] - dfA.fishing_trend.values[0]) / dt

# Outside China
nonfW_pre = dfW.nonfishing[pre].mean()
nonfW_pos = dfW.nonfishing[pos].mean()
nonfW_pre_sd = dfW.nonfishing[pre].std()
nonfW_pos_sd = dfW.nonfishing[pos].std()
nonfW_mean = dfW.nonfishing.mean()
nonfW_trend = (dfW.nonfishing_trend.values[-1] - dfW.nonfishing_trend.values[0]) / dt

# Inside China
nonfC_pre = dfC.nonfishing[pre].mean()
nonfC_pos = dfC.nonfishing[pos].mean()
nonfC_pre_sd = dfC.nonfishing[pre].std()
nonfC_pos_sd = dfC.nonfishing[pos].std()
nonfC_mean = dfC.nonfishing.mean()
nonfC_trend = (dfC.nonfishing_trend.values[-1] - dfC.nonfishing_trend.values[0]) / dt

# All
nonfA_pre = dfA.nonfishing[pre].mean()
nonfA_pos = dfA.nonfishing[pos].mean()
nonfA_pre_sd = dfA.nonfishing[pre].std()
nonfA_pos_sd = dfA.nonfishing[pos].std()
nonfA_mean = dfA.nonfishing.mean()
nonfA_trend = (dfA.nonfishing_trend.values[-1] - dfA.nonfishing_trend.values[0]) / dt

fishW_change1 = 100 * (fishW_pos - fishW_pre) / np.abs(fishW_pre)
fishC_change1 = 100 * (fishC_pos - fishC_pre) / np.abs(fishC_pre)
fishA_change1 = 100 * (fishA_pos - fishA_pre) / np.abs(fishA_pre)

fishW_change2 = 100 * (fishW_pos - fishW_mean) / np.abs(fishW_mean)
fishC_change2 = 100 * (fishC_pos - fishC_mean) / np.abs(fishC_mean)
fishA_change2 = 100 * (fishA_pos - fishA_mean) / np.abs(fishA_mean)

nonfW_change1 = 100 * (nonfW_pos - nonfW_pre) / np.abs(nonfW_pre)
nonfC_change1 = 100 * (nonfC_pos - nonfC_pre) / np.abs(nonfC_pre)
nonfA_change1 = 100 * (nonfA_pos - nonfA_pre) / np.abs(nonfA_pre)

nonfW_change1_sd = 100 * np.sqrt(nonfW_pos_sd**2 + nonfW_pre_sd**2)
nonfC_change1_sd = 100 * np.sqrt(nonfC_pos_sd**2 + nonfC_pre_sd**2)
nonfA_change1_sd = 100 * np.sqrt(nonfA_pos_sd**2 + nonfA_pre_sd**2)

nonfW_change2 = 100 * (nonfW_pos - nonfW_mean) / np.abs(nonfW_mean)
nonfC_change2 = 100 * (nonfC_pos - nonfC_mean) / np.abs(nonfC_mean)
nonfA_change2 = 100 * (nonfA_pos - nonfA_mean) / np.abs(nonfA_mean)

# ----- pre and post 2000 values ----- #
print("N pre and pos:", n_pre, n_pos)
print()

print("Fishing:  M_pre  SD_pre  M_pos  SD_pos")
print(f"World:   {fishW_pre:.1f}  {fishW_pre_sd:.1f}  {fishW_pos:.1f}  {fishW_pos_sd:.1f}")
print(f"China:   {fishC_pre:.1f}  {fishC_pre_sd:.1f}  {fishC_pos:.1f}  {fishC_pos_sd:.1f}")
print(f"Outside: {fishA_pre:.1f}  {fishA_pre_sd:.1f}  {fishA_pos:.1f}  {fishA_pos_sd:.1f}")
print()
print("Non-fishing: M_pre  SD_pre  M_pos  SD_pos")
print(f"World:   {nonfW_pre:.1f}  {nonfW_pre_sd:.1f}  {nonfW_pos:.1f}  {nonfW_pos_sd:.1f}")
print(f"China:   {nonfC_pre:.1f}  {nonfC_pre_sd:.1f}  {nonfC_pos:.1f}  {nonfC_pos_sd:.1f}")
print(f"Outside: {nonfA_pre:.1f}  {nonfA_pre_sd:.1f}  {nonfA_pos:.1f}  {nonfA_pos_sd:.1f}")
print()

print("Change due to COVID-19:\n")
print(f"{'':<18}2018-2019 vs 2020-2021 | 2017-2021 vs 2020-2021\n")

print(f"Outside China fishing change:{'':>5}{fishW_change1:+.1f}% | {fishW_change2:+.1f}%")
print(f"Inside China fishing change: {'':>6}{fishC_change1:+.1f}% | {fishC_change2:+.1f}%")
print(f"ALL activity fishing change:{'':>6}{fishA_change1:+.1f}% | {fishA_change2:+.1f}%")
print()
print(f"Outside China non-fishing change:{'':>2}{nonfW_change1:+.1f}% | {nonfW_change2:+.1f}%")
print(f"Inside China non-fishing change: {'':>2}{nonfC_change1:+.1f}% | {nonfC_change2:+.1f}%")
print(f"ALL activity non-fishing change: {'':>2}{nonfA_change1:+.1f}% | {nonfA_change2:+.1f}%")

print("\nTrends (2017-2021):\n")

print(f"Outside China fishing trend:{'':>6}{fishW_trend:+.0f}")
print(f"Inside China fishing trend: {'':>6}{fishC_trend:+.0f}")

print(f"Outside China non-fishing trend:{'':>2}{nonfW_trend:+.0f}")
print(f"Inside China non-fishing trend: {'':>2}{nonfC_trend:+.0f}")

# +
SAVE = False
FONT = 10

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams.update({"figure.autolayout": True})

fig = plt.figure(figsize=(7.0, 8.5), tight_layout=True)

# Define plotting grid

n = 2  # number of double-rows
m = 1  # number of columns

t = 0.95  # 1-t == top space
b = 0.09  # bottom space      (both in figure coordinates)

msp = 0.2  # minor spacing
sp = 0.4  # major spacing

offs = (1 + msp) * (t - b) / (2 * n + n * msp + (n - 1) * sp)  # grid offset
hspace = sp + msp + 1  # height space per grid

gso = GridSpec(
    n,
    m,
    bottom=b + offs,
    top=t,
    hspace=hspace,
    height_ratios=[1.1, 1.1],
    right=0.96,
)
gse = GridSpec(
    n,
    m,
    bottom=b,
    top=t - offs,
    hspace=hspace,
    height_ratios=[1.1, 1.19],
    right=0.96,
)

ax1a = fig.add_subplot(gso[0, :])
ax1b = fig.add_subplot(gse[0, :])
ax2a = fig.add_subplot(gso[1, :])
ax2b = fig.add_subplot(gse[1, :])

names = ["outside\nChina", "inside\nChina"]

color_fish = palette2[1]
color_nonf = "tab:blue"

text_loc = (0.02, 0.9)

# covid_fish = [f"${fishW_change1:+.0f}$%", f"${fishC_change1:+.0f}$%"]
# covid_nonf = [f"${nonfW_change1:+.0f}$%", f"${nonfC_change1:+.0f}$%"]
covid_fish = ["$-14 \pm 2$%", "$-8 \pm 3$%"]
covid_nonf = ["$-1 \pm 1$%", "$+4 \pm 1$%"]

# ----- Fishing ----- #

for ax, d, n, c in zip([ax1a, ax1b], [dfW, dfC], names, covid_fish):

    d = d.rolling(3, center=True).median()

    cond1 = (d.date_year > 2017) & (d.date_year < 2020)
    cond2 = (d.date_year > 2020) & (d.date_year < 2022)
    mean1 = d.fishing[cond1].mean()
    mean2 = d.fishing[cond2].mean()
    std1 = d.fishing[cond1].std()
    std2 = d.fishing[cond2].std()
    vmin = round(d.fishing.min(), -2)
    vmax = round(d.fishing.max(), -2)
    vmean = round(d.fishing.mean(), -2)
    valid = d.nonfishing_trend.notnull()
    vfirst, vlast = d.fishing_trend[valid].values[[0, -1]]
    perc = 100 * (vlast - vfirst) / vfirst
    print(vfirst, vlast, vmean)

    x = [2018, 2020, 2020, 2022]
    y1 = [mean1 + std1, mean1 + std1, mean2 + std2, mean2 + std2]
    y2 = [mean1 - std1, mean1 - std1, mean2 - std2, mean2 - std2]

    # Mean boxes
    ax.fill_between(x, y1=y1, y2=y2, color="0.925")
    ax.plot(x, y1, linewidth=0.5, color="0.75")
    ax.plot(x, y2, linewidth=0.5, color="0.75")
    
    # Mean line 
    ax.hlines(
        vmean,
        2017,
        2022,
        color=color_fish,
        linewidth=1,
        clip_on=True,
        linestyle="--",
    )

    # Time series
    ax.plot(
        d.date_year, d.fishing, linewidth=1.75, color=color_fish, clip_on=False
    )
    ax.set_ylim(d.fishing.min(), d.fishing.max())

    if "outside" in n.lower():
        
        # Pandemic line
        ax.hlines(
            vmax,
            2020,
            2022,
            color="0.1",
            linewidth=1.5,
            zorder=10,
            clip_on=True,
        )
        # Pandemic text
        ax.text(
            2021,
            vmax,
            "Pandemic",
            ha="center",
            va="center",
            color="0.1",
            fontsize=FONT,
            zorder=11,
            bbox=dict(facecolor="white", edgecolor="white"),
        )

    # Legend
    ax.text(
        *text_loc,
        n,
        ha="left",
        va="center",
        color=".1",
        fontsize=FONT,
        transform=ax.transAxes,
    )
    
    # Mean percent (top of box)
    ax.text(
        2020.08,
        (mean2 + std2) + 300,
        c,
        ha="left",
        va="bottom",
        color="0.1",
        fontsize=FONT,
        weight="normal",
    )

    ax.set_ylim(vmin, vmax)
    ax.set_yticks([vmin, vmean, vmax])

    if "outside" in n.lower():
        ax.set_yticklabels(
            [f"Min\n{vmin:.0f}", f"MEAN\n{vmean:.0f}", f"Max\n{vmax:.0f}"]
        )
        
    # ----- Mark cycles ----- #
    
    ys = np.arange(2017, 2023)
    nyw = ys - 0.05
    
    ys = np.arange(2017, 2022)
    nyc = ys + 0.05
    mor = ys + 0.325
        
    if "outside" in n.lower():
        for ny in nyw:
            ax.hlines(vmin, ny, ny+0.1, color='g', lw=5, alpha=0.5, clip_on=False)
            
    if "inside" in n.lower():
        for ny, mo in zip(nyc, mor):
            ax.hlines(vmin, ny, ny+0.1, color='r', lw=5, alpha=0.5, clip_on=False)
            ax.hlines(vmin, mo, mo+0.275, color='b', lw=5, alpha=0.5, clip_on=False)
        

# ----- Non-fishing ----- #

for ax, d, n, c in zip([ax2a, ax2b], [dfW, dfC], names, covid_nonf):

    d = d.rolling(3, center=True).median()

    cond1 = (d.date_year > 2017) & (d.date_year < 2020)
    cond2 = (d.date_year > 2020) & (d.date_year < 2022)
    mean1 = d.nonfishing[cond1].mean()
    mean2 = d.nonfishing[cond2].mean()
    std1 = d.nonfishing[cond1].std()
    std2 = d.nonfishing[cond2].std()
    vmin = round(d.nonfishing.min(), -2)
    vmax = round(d.nonfishing.max(), -2)
    vmean = round(d.nonfishing.mean(), -2)
    valid = d.nonfishing_trend.notnull()
    vfirst, vlast = d.nonfishing_trend[valid].values[[0, -1]]
    perc = 100 * (vlast - vfirst) / vfirst

    x = [2018, 2020, 2020, 2022]
    y1 = [mean1 + std1, mean1 + std1, mean2 + std2, mean2 + std2]
    y2 = [mean1 - std1, mean1 - std1, mean2 - std2, mean2 - std2]

    ax.fill_between(x, y1=y1, y2=y2, color="0.925")
    ax.plot(x, y1, linewidth=0.5, color="0.75")
    ax.plot(x, y2, linewidth=0.5, color="0.75")
    
    # Mean line 
    ax.hlines(
        vmean,
        2017,
        2022,
        color=color_nonf,
        linewidth=1,
        clip_on=True,
        linestyle="--",
    )

    ax.plot(
        d.date_year,
        d.nonfishing,
        linewidth=1.5,
        color=color_nonf,
        clip_on=False,
    )
    ax.set_ylim(d.nonfishing.min(), d.nonfishing.max())

    ax.text(
        *text_loc,
        n,
        ha="left",
        va="center",
        color=".1",
        fontsize=FONT,
        transform=ax.transAxes,
    )
    
    ax.text(
        2020.08,
        (mean2 + std2) + 100,
        c,
        ha="left",
        va="bottom",
        color="0.1",
        fontsize=FONT,
        weight="normal",
    )

    ax.set_ylim(vmin, vmax)
    ax.set_yticks([vmin, vmean, vmax])
    
    # ----- Mark cycles ----- #
    
    ys = np.arange(2017, 2023)
    nyw = ys - 0.05
    
    ys = np.arange(2017, 2022)
    nyc = ys + 0.05
    mor = ys + 0.325
        
    if "outside" in n.lower():
        for ny in nyw:
            ax.hlines(vmin, ny, ny+0.1, color='g', lw=5, alpha=0.5, clip_on=False)
            
    if "inside" in n.lower():
        for ny, mo in zip(nyc, mor):
            ax.hlines(vmin, ny, ny+0.1, color='r', lw=5, alpha=0.5, clip_on=False)
            ax.hlines(vmin, mo, mo+0.275, color='b', lw=5, alpha=0.5, clip_on=False)
            
# ----- Figure legend ----- #

y0 = vmin-1650

x0 = 2017.00
x1 = 2018.53
x2 = 2019.93
x3 = 2021.44

d0 = 0.1
ax2b.hlines(y0, x0, x0+d0, color='g', lw=10, alpha=0.5, clip_on=False)
ax2b.hlines(y0, x1, x1+d0, color='r', lw=10, alpha=0.5, clip_on=False)
ax2b.hlines(y0, x2, x2+d0, color='b', lw=10, alpha=0.5, clip_on=False)
ax2b.hlines(y0, x3, x3+d0, color='k', lw=10, alpha=0.3, clip_on=False)

ss = [
    "Christmas/New Year",
    "Chinese New Year",
    "Chinese Moratorium",
    "Mean $\pm$ SD",
]
for x, s in zip([x0, x1, x2, x3], ss):
    ax2b.text(
        x+d0+0.05,
        y0,
        s,
        ha="left",
        va="center",
        color='0.2',
        fontsize=FONT,
        clip_on=False,
    )

# ----- Figure legend ----- #

ax1a.text(
    0.02,
    1.05,
    "Industrial Fishing",
    ha="left",
    color=color_fish,
    weight="bold",
    fontsize=FONT,
    transform=ax1a.transAxes,
)
ax2a.text(
    0.02,
    1.05,
    "Transport and Energy",
    ha="left",
    color=color_nonf,
    weight="bold",
    fontsize=FONT,
    transform=ax2a.transAxes,
)

fig.text(
    0.03,
    0.52,
    "Number of detected vessels in the ocean",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=FONT,
)


ax1a.text(-0.08, 1.2, "a", fontsize=FONT+1, weight='bold', ha='left', va='top', transform=ax1a.transAxes)
ax2a.text(-0.08, 1.2, "b", fontsize=FONT+1, weight='bold', ha='left', va='top', transform=ax2a.transAxes)
             

# Hide the right and top spines
for ax in [ax1a, ax1b, ax2a, ax2b]:
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.minorticks_off()
    ax.grid(False)

for ax in [ax1a, ax2a]:
    ax.spines.bottom.set_visible(False)
    ax.set_xticks([])

fig.align_labels()

if SAVE:
    plt.savefig(
        "fishing_nonfishing_series_v3.jpg",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=300
    )
# -

"""
# ALL ACTIVITY

SAVE = False

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams.update({"figure.autolayout": True})

fig = plt.figure(figsize=(7.0, 8.5), tight_layout=True)

# Define plotting grid

n = 2  # number of double-rows
m = 1  # number of columns

t = 0.95  # 1-t == top space
b = 0.09  # bottom space      (both in figure coordinates)

msp = 0.2  # minor spacing
sp = 0.4  # major spacing

offs = (1 + msp) * (t - b) / (2 * n + n * msp + (n - 1) * sp)  # grid offset
hspace = sp + msp + 1  # height space per grid

gso = GridSpec(
    n,
    m,
    bottom=b + offs,
    top=t,
    hspace=hspace,
    height_ratios=[1.1, 1.1],
    right=0.96,
)
gse = GridSpec(
    n,
    m,
    bottom=b,
    top=t - offs,
    hspace=hspace,
    height_ratios=[1.1, 1.19],
    right=0.96,
)

ax1a = fig.add_subplot(gso[0, :])
ax1b = fig.add_subplot(gse[0, :])
ax2a = fig.add_subplot(gso[1, :])
ax2b = fig.add_subplot(gse[1, :])

names = ["All\nActivity", "inside\nChina"]

color_fish = palette2[1]
color_nonf = "tab:blue"

text_loc = (0.02, 0.9)

covid_fish = [f"${fishA_change1:+.0f}$%", f"${fishC_change1:+.0f}$%"]
covid_nonf = [f"${nonfA_change1:+.0f}$%", f"${nonfC_change1:+.0f}$%"]

# ----- Fishing ----- #

for ax, d, n, c in zip([ax1a, ax1b], [dfA, dfC], names, covid_fish):

    d = d.rolling(3, center=True).median()

    cond1 = (d.date_year > 2017) & (d.date_year < 2020)
    cond2 = (d.date_year > 2020) & (d.date_year < 2022)
    mean1 = d.fishing[cond1].mean()
    mean2 = d.fishing[cond2].mean()
    std1 = d.fishing[cond1].std()
    std2 = d.fishing[cond2].std()
    vmin = round(d.fishing.min(), -2)
    vmax = round(d.fishing.max(), -2)
    vmean = round(d.fishing.mean(), -2)
    valid = d.nonfishing_trend.notnull()
    vfirst, vlast = d.fishing_trend[valid].values[[0, -1]]
    perc = 100 * (vlast - vfirst) / vfirst
    print(vfirst, vlast, vmean)

    x = [2018, 2020, 2020, 2022]
    y1 = [mean1 + std1, mean1 + std1, mean2 + std2, mean2 + std2]
    y2 = [mean1 - std1, mean1 - std1, mean2 - std2, mean2 - std2]

    # Mean boxes
    ax.fill_between(x, y1=y1, y2=y2, color="0.925")
    ax.plot(x, y1, linewidth=0.5, color="0.75")
    ax.plot(x, y2, linewidth=0.5, color="0.75")
    
    # Mean line 
    ax.hlines(
        vmean,
        2017,
        2022,
        color=color_fish,
        linewidth=1,
        clip_on=True,
        linestyle="--",
    )

    # Time series
    ax.plot(
        d.date_year, d.fishing, linewidth=1.75, color=color_fish, clip_on=False
    )
    ax.set_ylim(d.fishing.min(), d.fishing.max())

    # ax.plot(
    #     d.date_year,
    #     d.fishing_trend,
    #     linewidth=1,
    #     color=color_fish,
    #     linestyle="--",
    # )

    if "all" in n.lower():
        
        # Pandemic line
        ax.hlines(
            vmax,
            2020,
            2022,
            color="0.5",
            linewidth=1.5,
            zorder=10,
            clip_on=True,
        )
        # Pandemic text
        ax.text(
            2021,
            vmax,
            "Pandemic",
            ha="center",
            va="center",
            color="0.5",
            fontsize=11,
            zorder=11,
            bbox=dict(facecolor="white", edgecolor="white"),
        )

    # Legend
    ax.text(
        *text_loc,
        n,
        ha="left",
        va="center",
        color=".1",
        fontsize=11,
        transform=ax.transAxes,
    )
    
    # Trend percent (right)
    # ax.text(
    #     2022.06,
    #     vlast,
    #     f"${perc:+.0f}$%",
    #     ha="left",
    #     va="center",
    #     color=color_fish,
    #     fontsize=11,
    #     weight="bold",
    # )
    
    # Mean percent (top of box)
    ax.text(
        2020.08,
        (mean2 + std2) + 300,
        c,
        ha="left",
        va="bottom",
        color="0.4",
        fontsize=11,
        weight="normal",
    )

    ax.set_ylim(vmin, vmax)
    ax.set_yticks([vmin, vmean, vmax])

    if "all" in n.lower():
        ax.set_yticklabels(
            [f"Min\n{vmin:.0f}", f"MEAN\n{vmean:.0f}", f"Max\n{vmax:.0f}"]
        )
        
    # ----- Mark cycles ----- #
    
    ys = np.arange(2017, 2023)
    nyw = ys - 0.05
    
    ys = np.arange(2017, 2022)
    nyc = ys + 0.05
    mor = ys + 0.325
        
    if "all" in n.lower():
        for ny in nyw:
            ax.hlines(vmin, ny, ny+0.1, color='g', lw=5, alpha=0.5, clip_on=False)
            
    if "inside" in n.lower():
        for ny, mo in zip(nyc, mor):
            ax.hlines(vmin, ny, ny+0.1, color='r', lw=5, alpha=0.5, clip_on=False)
            ax.hlines(vmin, mo, mo+0.275, color='b', lw=5, alpha=0.5, clip_on=False)
        
    if "all" in n.lower():
        ax.text(.01, .99, "a", fontsize=12, weight='bold', ha='left', va='top', transform=ax.transAxes)

# ----- Non-fishing ----- #

for ax, d, n, c in zip([ax2a, ax2b], [dfA, dfC], names, covid_nonf):

    d = d.rolling(3, center=True).median()

    cond1 = (d.date_year > 2017) & (d.date_year < 2020)
    cond2 = (d.date_year > 2020) & (d.date_year < 2022)
    mean1 = d.nonfishing[cond1].mean()
    mean2 = d.nonfishing[cond2].mean()
    std1 = d.nonfishing[cond1].std()
    std2 = d.nonfishing[cond2].std()
    vmin = round(d.nonfishing.min(), -2)
    vmax = round(d.nonfishing.max(), -2)
    vmean = round(d.nonfishing.mean(), -2)
    valid = d.nonfishing_trend.notnull()
    vfirst, vlast = d.nonfishing_trend[valid].values[[0, -1]]
    perc = 100 * (vlast - vfirst) / vfirst

    x = [2018, 2020, 2020, 2022]
    y1 = [mean1 + std1, mean1 + std1, mean2 + std2, mean2 + std2]
    y2 = [mean1 - std1, mean1 - std1, mean2 - std2, mean2 - std2]

    ax.fill_between(x, y1=y1, y2=y2, color="0.925")
    ax.plot(x, y1, linewidth=0.5, color="0.75")
    ax.plot(x, y2, linewidth=0.5, color="0.75")
    
    # Mean line 
    ax.hlines(
        vmean,
        2017,
        2022,
        color=color_nonf,
        linewidth=1,
        clip_on=True,
        linestyle="--",
    )

    ax.plot(
        d.date_year,
        d.nonfishing,
        linewidth=1.5,
        color=color_nonf,
        clip_on=False,
    )
    ax.set_ylim(d.nonfishing.min(), d.nonfishing.max())

    # ax.plot(
    #     d.date_year,
    #     d.nonfishing_trend,
    #     linewidth=1,
    #     color=color_nonf,
    #     linestyle="--",
    # )

    ax.text(
        *text_loc,
        n,
        ha="left",
        va="center",
        color=".1",
        fontsize=11,
        transform=ax.transAxes,
    )
    # ax.text(
    #     2022.06,
    #     vlast,
    #     f"${perc:+.0f}$%",
    #     ha="left",
    #     va="center",
    #     color=color_nonf,
    #     fontsize=11,
    #     weight="bold",
    # )
    ax.text(
        2020.08,
        (mean2 + std2) + 100,
        c,
        ha="left",
        va="bottom",
        color="0.4",
        fontsize=11,
        weight="normal",
    )

    ax.set_ylim(vmin, vmax)
    ax.set_yticks([vmin, vmean, vmax])
    
    # ----- Mark cycles ----- #
    
    ys = np.arange(2017, 2023)
    nyw = ys - 0.05
    
    ys = np.arange(2017, 2022)
    nyc = ys + 0.05
    mor = ys + 0.325
        
    if "all" in n.lower():
        for ny in nyw:
            ax.hlines(vmin, ny, ny+0.1, color='g', lw=5, alpha=0.5, clip_on=False)
            
    if "inside" in n.lower():
        for ny, mo in zip(nyc, mor):
            ax.hlines(vmin, ny, ny+0.1, color='r', lw=5, alpha=0.5, clip_on=False)
            ax.hlines(vmin, mo, mo+0.275, color='b', lw=5, alpha=0.5, clip_on=False)

    if "all" in n.lower():
        ax.text(.01, .99, "b", fontsize=12, weight='bold', ha='left', va='top', transform=ax.transAxes)
            
# ----- Figure legend ----- #

y0 = vmin-1650

x0 = 2017
x1 = 2018.50
x2 = 2019.64
x3 = 2020.96

d0 = 0.1
ax2b.hlines(y0, x0, x0+d0, color='g', lw=9, alpha=0.5, clip_on=False)
ax2b.hlines(y0, x1, x1+d0, color='r', lw=9, alpha=0.5, clip_on=False)
ax2b.hlines(y0, x2, x2+d0, color='b', lw=9, alpha=0.5, clip_on=False)
ax2b.hlines(y0, x3, x3+d0, color='k', lw=9, alpha=0.3, clip_on=False)

ss = [
    "Christmas/New Year",
    "Chinese New Year",
    "Chinese Moratorium",
    "Mean $\pm$ Std",
]
for x, s in zip([x0, x1, x2, x3], ss):
    ax2b.text(
        x+d0+0.05,
        y0,
        s,
        ha="left",
        va="center",
        color='0.2',
        fontsize=11,
        clip_on=False,
    )

# ----- Figure legend ----- #

ax1a.text(
    0.02,
    1.05,
    "Industrial Fishing",
    ha="left",
    color=color_fish,
    weight="bold",
    fontsize=11,
    transform=ax1a.transAxes,
)
ax2a.text(
    0.02,
    1.05,
    "Transport and Energy",
    ha="left",
    color=color_nonf,
    weight="bold",
    fontsize=11,
    transform=ax2a.transAxes,
)

fig.text(
    0.03,
    0.52,
    "Number of detected vessels in the ocean",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=11,
)

# Hide the right and top spines
for ax in [ax1a, ax1b, ax2a, ax2b]:
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.minorticks_off()
    ax.grid(False)

for ax in [ax1a, ax2a]:
    ax.spines.bottom.set_visible(False)
    ax.set_xticks([])

fig.align_labels()

if SAVE:
    plt.savefig(
        "./fishing_nonfishing_series_v3.png",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=300
    )
"""


