# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
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
# ## Time series of fishing and non-fishing only
# ### V8 (using rematched and interpolated data)

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

import pycountry

from pycountry_convert import (
    country_alpha2_to_continent_code,
    country_name_to_country_alpha2,
)

warnings.filterwarnings("ignore")
# %matplotlib inline

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
f = "../../data/24day_rolling_augmented_v20230220.feather"
df1 = pd.read_feather(f) #, parse_dates=["rolling_date"])

# %%

# %%
df1.rolling_date = df1.rolling_date.apply(lambda x: datetime.datetime(x.year,x.month,x.day))

# %%
# Crop beguining and end data points
cond1 = df1.rolling_date >= datetime.datetime(2017, 1, 13)
cond2 = df1.rolling_date < datetime.datetime(2021, 12, 20)
df1 = df1[cond1 & cond2]

# %%
print(df1.head())

# %%
dfW = df1[df1.eez_iso3 != "CHN"].groupby("rolling_date").sum()
dfC = df1[df1.eez_iso3 == "CHN"].groupby("rolling_date").sum()


# %%
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


for d in [dfW, dfC]:
    d = get_year_intervals(d)
    d = get_date_years(d)


# %%
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

for d in [dfW, dfC]:
    for calc_key in calc_keys:
        d = get_annual_means(d, calc_key)
        d = get_trend_and_cycle(d, calc_key)

# %%
# Calculate statistics

pre = (dfW.year_interval == "2018") | (dfW.year_interval == "2019")
post = (dfW.year_interval == "2020") | (dfW.year_interval == "2021")
dt = (dfW.date_year.values[-1] - dfW.date_year.values[0])  
# FIXME: double check the time interval

fishW_pre = dfW.fishing[pre].mean()
fishW_post = dfW.fishing[post].mean()
fishW_mean = dfW.fishing.mean()
fishW_trend = (dfW.fishing_trend.values[-1] - dfW.fishing_trend.values[0]) / dt

fishC_pre = dfC.fishing[pre].mean()
fishC_post = dfC.fishing[post].mean()
fishC_mean = dfC.fishing.mean()
fishC_trend = (dfC.fishing_trend.values[-1] - dfC.fishing_trend.values[0]) / dt

nonfW_pre = dfW.nonfishing[pre].mean()
nonfW_post = dfW.nonfishing[post].mean()
nonfW_mean = dfW.nonfishing.mean()
nonfW_trend = (dfW.nonfishing_trend.values[-1] - dfW.nonfishing_trend.values[0]) / dt

nonfC_pre = dfC.nonfishing[pre].mean()
nonfC_post = dfC.nonfishing[post].mean()
nonfC_mean = dfC.nonfishing.mean()
nonfC_trend = (dfC.nonfishing_trend.values[-1] - dfC.nonfishing_trend.values[0]) / dt

fishW_change1 = 100 * (fishW_post - fishW_pre) / np.abs(fishW_pre)
fishC_change1 = 100 * (fishC_post - fishC_pre) / np.abs(fishC_pre)
fishW_change2 = 100 * (fishW_post - fishW_mean) / np.abs(fishW_mean)
fishC_change2 = 100 * (fishC_post - fishC_mean) / np.abs(fishC_mean)

nonfW_change1 = 100 * (nonfW_post - nonfW_pre) / np.abs(nonfW_pre)
nonfC_change1 = 100 * (nonfC_post - nonfC_pre) / np.abs(nonfC_pre)
nonfW_change2 = 100 * (nonfW_post - nonfW_mean) / np.abs(nonfW_mean)
nonfC_change2 = 100 * (nonfC_post - nonfC_mean) / np.abs(nonfC_mean)


print("Change due to COVID-19:\n")
print(f"{'':<10}2018-2019 vs 2020-2021 | 2017-2021 vs 2020-2021\n")

print(f"World fishing change:{'':>5}{fishW_change1:+.1f}% | {fishW_change2:+.1f}%")
print(f"China fishing change:{'':>6}{fishC_change1:+.1f}% | {fishC_change2:+.1f}%")

print(f"World non-fishing change:{'':>2}{nonfW_change1:+.1f}% | {nonfW_change2:+.1f}%")
print(f"China non-fishing change:{'':>2}{nonfC_change1:+.1f}% | {nonfC_change2:+.1f}%")

print("\nTrends (2017-2021):\n")

print(f"World fishing trend:{'':>6}{fishW_trend:+.0f} units?")
print(f"China fishing trend:{'':>6}{fishC_trend:+.0f}")

print(f"World non-fishing trend:{'':>2}{nonfW_trend:+.0f}")
print(f"China non-fishing trend:{'':>2}{nonfC_trend:+.0f}")

# %%
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

names = ["outside\nChina", "inside\nChina"]

color_fish = palette2[1]
color_nonf = "tab:blue"

text_loc = (0.02, 0.9)

# Stats are calculated below
# # copy and paste numbers here
# covid_fish = ["$-14$%", "$-8$%"]
# covid_nonf = ["$<1$%", "$+5$%"]
covid_fish = [f"${fishW_change1:+.0f}$%", f"${fishC_change1:+.0f}$%"]
covid_nonf = [f"${nonfW_change1:+.0f}$%", f"${nonfC_change1:+.0f}$%"]

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
        color='0.5',
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

    if "outside" in n.lower():
        
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
        color="0.5",
        fontsize=11,
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

    ax.plot(
        d.date_year,
        d.nonfishing,
        linewidth=1.5,
        color=color_nonf,
        clip_on=False,
    )
    ax.set_ylim(d.nonfishing.min(), d.nonfishing.max())

    ax.plot(
        d.date_year,
        d.nonfishing_trend,
        linewidth=1,
        color=color_nonf,
        linestyle="--",
    )

    ax.text(
        *text_loc,
        n,
        ha="left",
        va="center",
        color=".1",
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.text(
        2022.06,
        vlast,
        f"${perc:+.0f}$%",
        ha="left",
        va="center",
        color=color_nonf,
        fontsize=11,
        weight="bold",
    )
    ax.text(
        2020.08,
        (mean2 + std2) + 100,
        c,
        ha="left",
        va="bottom",
        color="0.5",
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
        
    if "outside" in n.lower():
        for ny in nyw:
            ax.hlines(vmin, ny, ny+0.1, color='g', lw=5, alpha=0.5, clip_on=False)
            
    if "inside" in n.lower():
        for ny, mo in zip(nyc, mor):
            ax.hlines(vmin, ny, ny+0.1, color='r', lw=5, alpha=0.5, clip_on=False)
            ax.hlines(vmin, mo, mo+0.275, color='b', lw=5, alpha=0.5, clip_on=False)

            
# ----- Figure legend ----- #

y0 = vmin-1500

x0 = 2017
x1 = 2018.33
x2 = 2019.56
x3 = 2020.89

d0 = 0.1
ax2b.hlines(y0, x0, x0+d0, color='g', lw=7, alpha=0.5, clip_on=False)
ax2b.hlines(y0, x1, x1+d0, color='r', lw=7, alpha=0.5, clip_on=False)
ax2b.hlines(y0, x2, x2+d0, color='b', lw=7, alpha=0.5, clip_on=False)
ax2b.hlines(y0, x3, x3+d0, color='k', lw=7, alpha=0.3, clip_on=False)

ss = [
    "Christmas/new year",
    "Chinese new year",
    "Chinese moratorium",
    "Mean $\pm$ std",
]
for x, s in zip([x0, x1, x2, x3], ss):
    ax2b.text(
        x+d0+0.05,
        y0,
        s,
        ha="left",
        va="center",
        color='0.2',
        fontsize=9,
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
    plt.savefig("../../figures/fig3v2.png", bbox_inches="tight", pad_inches=0.1, dpi=300)

# %%
