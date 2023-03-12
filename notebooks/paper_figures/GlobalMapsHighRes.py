# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Make High-Resolution Global Maps of Vessel Detections from Sentinel-1
#
#

# %matplotlib inline
from datetime import datetime, timedelta

import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mpcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import proplot as pplt
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
import pyperclip

# +
# use the standard for eliminating ice locations.
from prj_global_sar_analysis.eliminate_ice_string import eliminate_ice_string

eliminated_locations = eliminate_ice_string()
pyperclip.copy(eliminated_locations)

# +
import pycountry
from pycountry_convert import (
    country_alpha2_to_continent_code,
    country_alpha2_to_country_name,
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


# -


# +
vessel_info_table = "gfw_research.vi_ssvid_v20221001"


predictions_table = """
select 
  detect_id,
  rf_fishing_low as fishing_score_low,
  rf_fishing fishing_score, 
  rf_fishing_high fishing_score_high
from 
  `world-fishing-827.scratch_pete.rf_nn_results_20221104`

"""
predictions_table = """
select 
  detect_id,
  rf_fishing_low/2 + nn_fishing_low/2 as fishing_score_low,
  rf_fishing/2 + nn_fishing/2 fishing_score, 
  rf_fishing_high/2 + nn_fishing_high/2 fishing_score_high
from 
  `world-fishing-827.scratch_pete.rf_nn_results_20221104`

"""

predictions_table = """

  select 
    detect_id, 
    avg(fishing_33) fishing_score_low,
    avg( fishing_50) fishing_score, 
    avg(fishing_66) fishing_score_high
  from
    (select detect_id, fishing_33, fishing_50, fishing_66 from 
    `world-fishing-827.proj_sentinel1_v20210924.fishing_pred_even_v5*`
    union all
    select detect_id, fishing_33, fishing_50, fishing_66 from 
    `world-fishing-827.proj_sentinel1_v20210924.fishing_pred_odd_v5*`
    )
  group by 
    detect_id
"""

scale = 20
# -

q = f"""with
predictions_table as
(
{predictions_table}
),

vessel_info as (
select
  ssvid,
  if(on_fishing_list_known is not null, on_fishing_list_known, on_fishing_list_nn) as on_fishing_list
from
   `world-fishing-827.{vessel_info_table}`
  -- don't do anything with identity spoofing vessels!
  where activity.overlap_hours_multinames < 24
),

detections_table as

(
  select
  floor(detect_lat*5) as lat_index_d,
  floor(detect_lon*5) as lon_index_d,
  detect_lat,
  detect_lon,
  detect_id,
  ssvid,
  eez_iso3,
  score,
  confidence,
  overpasses_2017_2021,
  date_24
  from
  `world-fishing-827.proj_global_sar.detections_w_overpasses_v20220929`
  where
  -- the following is very restrictive on repeated objects
  repeats_100m_180days_forward < 3 and
  repeats_100m_180days_back < 3 and
  repeats_100m_180days_center < 3
  -- get rid of scenes where more than half the detections
  -- are likely noise
  and (scene_detections <=5 or scene_quality > .5)
  and extract(date from detect_timestamp)
     between "2017-01-01" and "2021-12-31"
  -- at least 10 overpasses
  and overpasses_2017_2021 > 10
  -- our cutoff for noise -- this could be adjusted down, but makes
  -- very little difference between .5 and .7
  and presence > .7
  {eliminated_locations}
  ),


dark_vessel_density as (
select 
  lat_index_d,
  lon_index_d,
  greatest(dd_perkm2, 1e-5) dd_perkm2
from
(
select
  lat_index_d,
  lon_index_d,
  sum(if(score < 1e-3,(1/overpasses_2017_2021) / pow(111/5 * cos((lat_index_d+.5)/5*3.1416/180),2),0)) dd_perkm2 -- dark detects per km2
from 
  detections_table
 -- this is very insensitive to the exact cut off
group by 
  lat_index_d, lon_index_d)
),




final_table as (
select
  lat_index_d,
  lon_index_d,
  date_24,
  detect_lat,
  detect_lon,
  overpasses_2017_2021,
  eez_iso3,
  fishing_score,
  fishing_score_low,
  fishing_score_high,
  case when score > dd_perkm2 and on_fishing_list and confidence > .5 then "matched_fishing"
   when score > dd_perkm2 and not on_fishing_list and confidence > .5 then "matched_nonfishing"
   when score > dd_perkm2 and ( on_fishing_list is null or confidence < .5) then "matched_unknown"
   when score < dd_perkm2 then "unmatched" end as matched_category
from
  detections_table a
left join
  vessel_info
using(ssvid)
left join
  predictions_table
using(detect_id)
left join
 dark_vessel_density
 using(lat_index_d, lon_index_d)

)

select
  floor(detect_lat*{scale}) lat_index,
  floor(detect_lon*{scale}) lon_index,
  eez_iso3,
  sum(if( matched_category = 'matched_fishing', 1/overpasses_2017_2021, 0)) matched_fishing,
  sum(if( matched_category = 'matched_nonfishing', 1/overpasses_2017_2021, 0)) matched_nonfishing,
  sum(if( matched_category = 'matched_unknown', 1/overpasses_2017_2021, 0)) matched_unknown,
  sum(if( matched_category = 'matched_unknown',
               fishing_score/overpasses_2017_2021, 0)) matched_unknown_likelyfish,
  sum(if( matched_category = 'matched_unknown',
               (1-fishing_score)/overpasses_2017_2021, 0)) matched_unknown_likelynonfish,
  sum(if( matched_category = 'unmatched', fishing_score/overpasses_2017_2021, 0)) unmatched_fishing,
  sum(if( matched_category = 'unmatched', (1-fishing_score)/overpasses_2017_2021, 0)) unmatched_nonfishing,
  
  sum(if( matched_category = 'matched_unknown',
               fishing_score_low/overpasses_2017_2021, 0)) matched_unknown_likelyfish_low,
  sum(if( matched_category = 'matched_unknown',
               (1-fishing_score_low)/overpasses_2017_2021, 0)) matched_unknown_likelynonfish_low,
  sum(if( matched_category = 'unmatched', 
               fishing_score_low/overpasses_2017_2021, 0)) unmatched_fishing_low,
  sum(if( matched_category = 'unmatched', 
                (1-fishing_score_low)/overpasses_2017_2021, 0)) unmatched_nonfishing_low,

  sum(if( matched_category = 'matched_unknown',
               fishing_score_high/overpasses_2017_2021, 0)) matched_unknown_likelyfish_high,
  sum(if( matched_category = 'matched_unknown',
               (1-fishing_score_high)/overpasses_2017_2021, 0)) matched_unknown_likelynonfish_high,
  sum(if( matched_category = 'unmatched', fishing_score_high/overpasses_2017_2021, 0)) unmatched_fishing_high,
  sum(if( matched_category = 'unmatched', (1-fishing_score_high)/overpasses_2017_2021, 0)) unmatched_nonfishing_high,
  
  
  sum(1/overpasses_2017_2021) detections
from
  final_table
group by
  eez_iso3, lat_index, lon_index"""


import pyperclip

pyperclip.copy(q)

df = pd.read_gbq(q)


df.head()

df["country"] = df.eez_iso3.apply(get_country)
df["continent"] = df.eez_iso3.apply(get_continent)
df["AIS fishing"] = df.matched_fishing + df.matched_unknown_likelyfish
df["AIS non-fishing"] = (
    df.matched_nonfishing + df.matched_unknown_likelynonfish
)
df["dark fishing"] = df.unmatched_fishing
df["dark fishing low"] = df.unmatched_fishing_low
df["dark fishing high"] = df.unmatched_fishing_high
df["dark non-fishing"] = df.unmatched_nonfishing
df["dark non-fishing low"] = df.unmatched_nonfishing_high
df["dark non-fishing high"] = df.unmatched_nonfishing_low
df["tot_fishing"] = df["dark fishing"] + df["AIS fishing"]
df["tot_nonfishing"] = df["dark non-fishing"] + df["AIS non-fishing"]

df.detections.sum()

df.tot_fishing.sum()

df.tot_nonfishing.sum()

df.tot_fishing.sum() / df.detections.sum()

# # Map It


# +


def map_bivariate(
    grid_total,
    grid_ratio,
    title,
    vmax=0.2,
    a_vmin=0.1,
    a_vmax=10,
    eez_linewidth=0.4,
    eez_color="#5c5c5c",
    filename="temp.png",
    savefig=False,
    figsize=(15, 9),
):

    cmap = psm.cm.bivariate.TransparencyBivariateColormap(
        psm.cm.bivariate.blue_orange
    )
    with psm.context(psm.styles.light):
        fig = plt.figure(figsize=figsize)
        ax = psm.create_map()
        #         psm.add_land(ax)
        #         psm.add_eezs()
        #         psm.add_eezs(ax, edgecolor=eez_color, linewidth=eez_linewidth)
        norm1 = mpcolors.Normalize(vmin=0.0, vmax=vmax, clip=True)
        norm2 = mpcolors.LogNorm(vmin=a_vmin, vmax=a_vmax, clip=True)

        psm.add_bivariate_raster(
            grid_ratio, grid_total, cmap, norm1, norm2, origin="lower"
        )

        psm.add_eezs()

        cb_ax = psm.add_bivariate_colorbox(
            cmap,
            norm1,
            norm2,
            xlabel="Fraction matched to AIS",
            ylabel="Detections/km$^{2}$",
            xformat="{x:.0%}",
            yformat="{x:.2f}",
            aspect_ratio=2.0,
            fontsize=12,
        )
        ax.set_title(title, pad=10, fontsize=15)

        if savefig:
            plt.savefig(filename, dpi=300, bbox_inches="tight")


# -

df.head()

# +
fishing_total = psm.rasters.df2raster(
    df,
    "lon_index",
    "lat_index",
    "tot_fishing",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

dark_fishing = psm.rasters.df2raster(
    df,
    "lon_index",
    "lat_index",
    "dark fishing",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

fishing_ratio = dark_fishing / fishing_total
# -

df.columns


map_bivariate(
    fishing_total,
    fishing_ratio,
    "Global Fishing Activity from SAR",
    vmax=1,
    a_vmin=0.00005,
    a_vmax=0.01,
    eez_linewidth=0.4,
    eez_color="#5c5c5c",
    filename="frac_dark_fishing_20th.png",
    savefig=True,
    figsize=(30, 20),
)


# +
nonfishing_total = psm.rasters.df2raster(
    df,
    "lon_index",
    "lat_index",
    "tot_nonfishing",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

dark_nonfishing = psm.rasters.df2raster(
    df,
    "lon_index",
    "lat_index",
    "dark non-fishing",
    xyscale=scale,
    per_km2=True,
    origin="lower",
)

nonfishing_ratio = dark_nonfishing / nonfishing_total

# +
map_bivariate(
    nonfishing_total,
    nonfishing_ratio,
    "Global Non-Fishing Activity from SAR",
    vmax=1,
    a_vmin=0.00005,
    a_vmax=0.01,
    eez_linewidth=0.4,
    eez_color="#5c5c5c",
    filename="frac_dark_nonfishing_20th.png",
    savefig=True,
    figsize=(30, 20),
)

# +
norm = mpcolors.LogNorm(vmin=0.00005, vmax=0.05)

with psm.context(psm.styles.light):
    fig = plt.figure(figsize=(30, 20))

    ax, im = psm.plot_raster(
        dark_fishing, cmap="fishing", norm=norm, origin="lower"
    )
    psm.add_land(ax)

    fig.colorbar(
        im, ax=ax, orientation="horizontal", fraction=0.02, aspect=40, pad=0.04
    )
    #     psm.add_eezs(ax, edgecolor=eez_color, linewidth=eez_linewidth)

    #     ax.set_title(title, pad=10, fontsize=15)
    #     ax.set_extent(map_extent)

    plt.savefig("dark_fishing_avg_20.png", dpi=300, bbox_inches="tight")

# +
norm = mpcolors.LogNorm(vmin=0.00005, vmax=0.05)

with psm.context(psm.styles.light):
    fig = plt.figure(figsize=(30, 20))

    ax, im = psm.plot_raster(
        fishing_total, cmap="fishing", norm=norm, origin="lower"
    )
    psm.add_land(ax)

    fig.colorbar(
        im, ax=ax, orientation="horizontal", fraction=0.02, aspect=40, pad=0.04
    )
    #     psm.add_eezs(ax, edgecolor=eez_color, linewidth=eez_linewidth)

    #     ax.set_title(title, pad=10, fontsize=15)
    #     ax.set_extent(map_extent)

    plt.savefig("tot_fishing_avg_20.png", dpi=300, bbox_inches="tight")
# -

df.head()

df.to_csv("20th_degree.csv", index=False)
