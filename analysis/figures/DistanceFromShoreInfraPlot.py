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
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
import proplot as pplt
import skimage
from datetime import datetime, timedelta
from dataclasses import dataclass
from matplotlib.gridspec import GridSpec

# %matplotlib inline

# %%
# Tableu 10
blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
red = '#d62728'
purple = '#9467bd'
brown = '#8c564b'
pink = '#e377c2'
gray = '#7f7f7f'
olive = '#bcbd22'
cyan = '#17becf'

# # Set1 colorbrewer
# red = '#e41a1c'
# blue = '#377eb8'
# green = '#4daf4a'
# purple = '#984ea3'
# orange = '#ff7f00'
yellow = '#ffff33'
# brown = '#a65628'
# pink = '#f781bf'

fuchsia = '#FF00FF'
ivory = '#FFFFCB'
pink2 = '#FFC0CB'
crimson = '#DC143C'
gold = '#FFD700'
blue2 = '#37A8B7'
blue2b = '#268EB5'
blue3 = '#131C69'
yellow2 = '#EAF9A4'

darker_blue = '#091346'
dark_blue = '#004a7b'
dark_green = '#008195'
inter_green = '#4bb69c'
light_green = '#bde6a5'
lighter_green = '#E1F5A3'
green_yellow = '#F3FCB4'

# %%
df_dist1 = pd.read_feather('../data/fishing_dist_from_infra.feather', use_threads=True)
df_dist2 = pd.read_feather('../data/distance_from_shore_vessel.feather', use_threads=True)
df_dist3 = pd.read_feather('../data/distance_from_shore_infra.feather', use_threads=True)

df_dist3.head()

# %%
SAVE = True
FONT = 8

scl = 0.583333333

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': FONT})

fig, axs = plt.subplots(1, 2, figsize=(12 * scl, 6 * scl))
ax3, ax4 = axs

# REVERSE PLOTS
ax4, ax3 = ax3, ax4

with psm.context(psm.styles.light):

    ### Distance from shore ###
        
    d2 = df_dist2[df_dist2.distance_from_shore_km >= 2]
    d3 = df_dist3[df_dist3.distance_from_shore_km >= 2]
    
    x_vessel, y_fish, y_nonf = d2.distance_from_shore_km, d2.tot_fishing, d2.tot_nonfishing
    x_infra, y_oil, y_wind = d3.distance_from_shore_km.to_numpy(), d3.oil.to_numpy(), d3.wind.to_numpy()
    
    color1='#091346'
    color2='#005c86'
    color3='#13a49a'
    color4='#bde6a5'
    label1 = 'Nonfishing vessels'
    label2 = 'Fishing vessels'
    label3 = 'Oil infrastructure'
    label4 = 'Wind infrastructure'
    
    ax3.plot(x_vessel, y_nonf, color=darker_blue, linewidth=2.25 * scl, linestyle='-', label=label1)
    ax3.plot(x_vessel, y_fish, color=darker_blue, linewidth=2.5 * scl, linestyle='--', label=label2)
    ax3.plot(x_infra, y_oil, color=inter_green, linewidth=2.25 * scl, linestyle='-', label=label3)
    ax3.plot(x_infra, y_wind, color=inter_green, linewidth=2.5 * scl, linestyle='--', label=label4)
    
    ax3.set_xlim(0, 60)
    ax3.set_xlabel("Distance from shore, km")
    ax3.set_ylabel("Average number of vessels or structures per day")
    ax3.legend(frameon=False, fontsize=FONT, loc='center right')
        
    ### Distance from infrastructure ###
    
    for label in ['oil', 'wind']:
        
        # trawlers
        d1 = df_dist1[(df_dist1.geartype == 'trawlers') & (df_dist1.label == label)]
        x1 = d1.distance_m/1000
        y1 = d1.normalized_fishing_activity.rolling(3).mean()/d1.normalized_fishing_activity.sum()
        
        # others
        d2 = df_dist1[(df_dist1.geartype != 'trawlers') & (df_dist1.label == label)].groupby(['distance_m','label']).sum().reset_index()
        x2 = d2.distance_m/1000
        y2 = d2.normalized_fishing_activity.rolling(3).mean()/d2.normalized_fishing_activity.sum()
    
        if label == 'oil':
            style = '-'
            width = 2.5 * scl
            label = 'From oil'
        elif label == 'wind':
            style = '--'
            width = 1.5 * scl
            label = 'From wind'
        
        label1 = label + ' (trawlers)'
        label2 = label + ' (other)'
            
        ax4.plot(x1, y1, color=darker_blue, linewidth=width, linestyle=style, label=label1)
        ax4.plot(x2, y2, color=inter_green, linewidth=width, linestyle=style, label=label2)        
        
        ax4.set_xlim(0, 6)
        ax4.set_xlabel("Distance from infrastructure, km")
        ax4.set_ylabel("Normalized relative amount of activity")
        ax4.legend(frameon=False, fontsize=FONT, loc='lower right')
        
    for ax in [ax3, ax4]:
        # Hide the right and top spines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.minorticks_off()
        # ax.grid(axis="y")
        ax.grid(False)
        
    # REVERSED SUBPLOTS
    ax4.text(-0.18, 1.03, 'a', fontsize=FONT+1, weight='bold', ha='left', va='top', transform=ax4.transAxes)        
    ax3.text(-0.18, 1.03, 'b', fontsize=FONT+1, weight='bold', ha='left', va='top', transform=ax3.transAxes)        

if SAVE:
    plt.savefig(
        "distance_from_shore_infra.jpg",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )

# %%
