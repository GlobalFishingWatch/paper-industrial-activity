# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
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
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
# %matplotlib inline

q = '''SELECT 
_partitiontime date,
sum(cos(lat_index/10*3.14/180)*111*111*.1*.1) area
FROM `world-fishing-827.proj_sentinel1_v20210924.detect_foot_raster_10` 
group by date order by date'''
df = pd.read_gbq(q)

plt.figure(figsize=(8,5))
plt.plot(df.date, df.area.rolling(12).mean())
plt.ylim(0,1.5e7)
plt.title("Area of Ocean Imaged each Day by Sentinel-1\n12 day rolling average")
plt.ylabel("area per day, square km")

plt.figure(figsize=(8,5))
plt.plot(df.date, df.area)
plt.ylim(0,1.5e7)
plt.title("Area of Ocean Imaged each Day by Sentinel-1")
plt.ylabel("area per day, square km")


