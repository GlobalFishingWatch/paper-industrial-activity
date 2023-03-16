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
# # DELETE THIS FILE???? or is it used??? 

# %%
# %matplotlib inline
from datetime import datetime, timedelta

# %%
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# %%
import numpy as np
import pandas as pd
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
import proplot as pplt

# %%
# use the standard for eliminating ice locations.
from prj_global_sar_analysis.eliminate_ice_string import eliminate_ice_string

eliminated_locations = eliminate_ice_string()

# %% [markdown]
# ## Map global fishing activity

# %%
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
):

    cmap = psm.cm.bivariate.TransparencyBivariateColormap(
        psm.cm.bivariate.blue_orange
    )
    with psm.context(psm.styles.light):
        fig = plt.figure(figsize=(15, 9))
        ax = psm.create_map()
        #         psm.add_land(ax)
        #         psm.add_eezs()
        #         psm.add_eezs(ax, edgecolor=eez_color, linewidth=eez_linewidth)
        norm1 = mpcolors.Normalize(vmin=0.0, vmax=vmax, clip=True)
        norm2 = mpcolors.LogNorm(vmin=a_vmin, vmax=a_vmax, clip=True)

        psm.add_bivariate_raster(
            grid_ratio, grid_total, cmap, norm1, norm2, origin="lower"
        )

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


# %%
vessel_info_table = "gfw_research.vi_ssvid_v20221101"
rf_prediction_table = "proj_global_sar.rf_predictions_calibrated_v20220912_"

scale = 5

# %%
q = f"""with

predictions_table as
(
  select 
    detect_id, fishing as fishing_score 
  from
    `world-fishing-827.{rf_prediction_table}*` 
  where 
    _table_suffix between "20170101" and "20211231"
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
  territory1,
  score,
  confidence,
  overpasses_2017_2021,  
  date_24
  from
  proj_global_sar.detections_w_overpasses_v20220805
  left join `world-fishing-827.gfw_research.eez_info`
  on (eez_iso3 = territory1_iso3)
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
 sum(1/overpasses_2017_2021) / pow(111/5 * cos((lat_index_d+.5)/5*3.1416/180),2) dd_perkm2 -- dark detects per km2
from detections_table
where score < 1e-3 -- this is very insensitive to the exact cut off
group by lat_index_d, lon_index_d
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
  territory1,
  fishing_score,
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
  territory1,
  sum(if( matched_category = 'matched_fishing', 1/overpasses_2017_2021, 0)) matched_fishing,
  sum(if( matched_category = 'matched_nonfishing', 1/overpasses_2017_2021, 0)) matched_nonfishing,
  sum(if( matched_category = 'matched_unknown', 1/overpasses_2017_2021, 0)) matched_unknown,
  sum(if( matched_category = 'matched_unknown' and fishing_score >= .5,
               1/overpasses_2017_2021, 0)) matched_unknown_likelyfish,
  sum(if( matched_category = 'matched_unknown' and fishing_score < .5,
               1/overpasses_2017_2021, 0)) matched_unknown_likelynonfish,               
  sum(if( matched_category = 'unmatched' and fishing_score >= .5, 1/overpasses_2017_2021, 0)) unmatched_fishing_t,
  sum(if( matched_category = 'unmatched' and fishing_score < .5, 1/overpasses_2017_2021, 0)) unmatched_nonfishing_t,
  sum(if( matched_category = 'unmatched' and fishing_score >= .8, 1/overpasses_2017_2021, 0)) unmatched_fishing_t_8,
  sum(if( matched_category = 'unmatched' and fishing_score < .8, 1/overpasses_2017_2021, 0)) unmatched_nonfishing_t_8,
  sum(if( matched_category = 'unmatched' and fishing_score >= .2, 1/overpasses_2017_2021, 0)) unmatched_fishing_t_2,
  sum(if( matched_category = 'unmatched' and fishing_score < .2, 1/overpasses_2017_2021, 0)) unmatched_nonfishing_t_2,
  sum(if( matched_category = 'unmatched', fishing_score/overpasses_2017_2021, 0)) unmatched_fishing_prob,
  sum(if( matched_category = 'unmatched', (1-fishing_score)/overpasses_2017_2021, 0)) unmatched_nonfishing_prob,
  sum(1/overpasses_2017_2021) detections
from 
  final_table
group by 
  eez_iso3, territory1, lat_index, lon_index"""

# %%
df = pd.read_gbq(q)

# %%
# df['country'] = df.eez_iso3.apply(get_country)
# df['continent'] = df.eez_iso3.apply(get_continent)
df["AIS fishing"] = df.matched_fishing + df.matched_unknown_likelyfish
df["AIS non-fishing"] = (
    df.matched_nonfishing + df.matched_unknown_likelynonfish
)
df["dark fishing"] = df.unmatched_fishing_prob
df["dark non-fishing"] = df.unmatched_nonfishing_prob
df["dark fishing t"] = df.unmatched_fishing_t
df["dark non-fishing t"] = df.unmatched_nonfishing_t
df["tot_fishing"] = df["dark fishing"] + df["AIS fishing"]
df["tot_nonfishing"] = df["dark non-fishing"] + df["AIS non-fishing"]

# %%
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

# %%
map_bivariate(
    fishing_total,
    fishing_ratio,
    "Global Fishing Activity from SAR",
    vmax=1,
    a_vmin=0.00005,
    a_vmax=0.01,
    eez_linewidth=0.4,
    eez_color="#5c5c5c",
    filename="frac_dark_fishing.png",
    savefig=True,
)

# %%
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

# %%
map_bivariate(
    nonfishing_total,
    nonfishing_ratio,
    "Global Non-Fishing Activity from SAR",
    vmax=1,
    a_vmin=0.00005,
    a_vmax=0.01,
    eez_linewidth=0.4,
    eez_color="#5c5c5c",
    filename="frac_dark_nonfishing.png",
    savefig=True,
)

# %% [markdown]
# ## Infrastructure

# %%
q = f"""
SELECT 
  lon,
  lat,
  label
FROM 
  `world-fishing-827.proj_global_sar.infrastructure_reclassified_v20221206`
"""
df = pd.read_gbq(q)

# %%
df.head()

# %%
df.label.unique()

# %%
len(df)
df.to_feather("../../data/oil_w_polygons.feather")

# %% [markdown]
# ### Make a map of detections by class

# %%
other = df.loc[df["label"] == "other"]

wind = df.loc[df["label"] == "wind"]

oil = df.loc[df["label"] == "oil"]

# %%
with psm.context(psm.styles.light):
    fig = plt.figure(figsize=(20, 10))
    psm.create_map()
    psm.add_land()
    psm.add_eezs()

    plt.scatter(
        oil.lon, oil.lat, s=5, color="g", transform=psm.identity, label="Oil"
    )

    plt.scatter(
        wind.lon,
        wind.lat,
        s=5,
        color="r",
        transform=psm.identity,
        label="Wind",
    )

    plt.scatter(
        other.lon,
        other.lat,
        s=5,
        color="b",
        transform=psm.identity,
        label="Other",
    )

    plt.legend(fontsize=24, markerscale=6)
    plt.tight_layout()
    # plt.savefig('infra_map', dpi = 300)

# %%
