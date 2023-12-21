# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Area Imaged

import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import pandas as pd
import pyseas.maps as psm
import pyseas.contrib as psc
import pyseas.cm
import matplotlib as mpl
import proplot
import matplotlib.dates as mdates
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
# %matplotlib inline

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = "Roboto"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['figure.dpi'] = 600



q = '''SELECT 
_partitiontime date,
sum(cos(lat_index/10*3.14/180)*111*111*.1*.1) area
FROM `proj_sentinel1_v20210924.detect_foot_raster_10` 
group by date order by date'''
df = pd.read_gbq(q)

# +
fig, ax = plt.subplots(figsize=(9,5))
plt.plot(df.date.values, df.area.rolling(12).mean().values)
plt.ylim(0,1.5e7)
# plt.title("Area of ocean imaged each day by Sentinel-1\n12 day rolling average")
plt.ylabel("Area per day, square km", fontsize=12)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# plt.savefig("figures/AreaImaged.jpeg",dpi = 300, bbox_inches="tight",facecolor="white")

plt.savefig("figures/AreaImaged.pdf",
    transparent=True,
    bbox_inches="tight",
    pad_inches=0.1,
    dpi='figure',
    facecolor="white")
# -

plt.figure(figsize=(8,5))
plt.plot(df.date.values, df.area.values)
plt.ylim(0,1.5e7)
plt.title("Area of ocean imaged each day by sentinel-1")
plt.ylabel("area per day, square km")


