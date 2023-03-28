# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MakeContinentBarChart
#
# This notebook produces the bar chart that is used on Figure 1 (fishing and non-fishing vessels by continent) and also produces the supplemental figure that provides the number of vessels by EEZ.
#

# %%
# %matplotlib inline
from datetime import datetime, timedelta

# %%
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mpcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# %%
import numpy as np
import pandas as pd
import proplot as pplt
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
import pyperclip
import seaborn as sns

# %%
# use the standard for eliminating ice locations.
from analyses_module.eliminate_ice_string import eliminate_ice_string

eliminated_locations = eliminate_ice_string()
pyperclip.copy(eliminated_locations)

# %%
import pycountry
from pycountry_convert import (
    country_alpha2_to_continent_code,
    country_alpha2_to_country_name,
    country_name_to_country_alpha2,
)

continents = {
    'NA': 'North America',
    'SA': 'South America',
    'AS': 'Asia',
    'OC': 'Australia',
    'AF': 'Africa',
    'EU': 'Europe'
}

def get_continent(x):
    try:
        return continents[country_alpha2_to_continent_code(pycountry.countries.get(alpha_3=x).alpha_2)]
    except:
        "None"

def get_country(x):
    try:
        return country_alpha2_to_country_name(pycountry.countries.get(alpha_3=x).alpha_2)
    except:
        "None"

# %%
vessel_info_table = "gfw_research.vi_ssvid_v20221001"

predictions_table = '''
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
    '''

scale = 5

# %%
q = f'''with
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
  detect_lat,
  detect_lon,
  detect_id,
  ssvid_mult_recall_length as ssvid,
  eez_iso3,
  score_mult_recall_length as score,
  7.4e-6 as dd_perkm2,
  overpasses_2017_2021,
  date_24
  from
  `world-fishing-827.proj_global_sar.detections_w_overpasses_v20230215`
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
  and overpasses_2017_2021 > 30
  -- our cutoff for noise -- this could be adjusted down, but makes
  -- very little difference between .5 and .7
  and presence > .7
  and not in_road_doppler
  {eliminated_locations}
  ),
  
final_table as (
select
  date_24,
  detect_lat,
  detect_lon,
  overpasses_2017_2021,
  eez_iso3,
  fishing_score,
  fishing_score_low,
  fishing_score_high,
  case when score > dd_perkm2 and on_fishing_list then "matched_fishing"
   when score > dd_perkm2 and not on_fishing_list then "matched_nonfishing"
   when score > dd_perkm2 and on_fishing_list is null then "matched_unknown"
   when score < dd_perkm2 then "unmatched" end as matched_category
from
  detections_table a
left join
  vessel_info
using(ssvid)
left join
  predictions_table
using(detect_id)
)

select
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
  eez_iso3'''


# %%
df = pd.read_gbq(q)

# %%
df.head()

# %%
df['country'] = df.eez_iso3.apply(get_country)
df['continent'] = df.eez_iso3.apply(get_continent)
df['AIS fishing'] = df.matched_fishing + df.matched_unknown_likelyfish
df['AIS non-fishing'] = df.matched_nonfishing + df.matched_unknown_likelynonfish
df['dark fishing'] = df.unmatched_fishing
df['dark fishing low'] = df.unmatched_fishing_low
df['dark fishing high'] = df.unmatched_fishing_high
df['dark non-fishing'] = df.unmatched_nonfishing
df['dark non-fishing low'] = df.unmatched_nonfishing_high
df['dark non-fishing high'] = df.unmatched_nonfishing_low
df['tot_fishing'] = df['dark fishing'] + df['AIS fishing']
df['tot_nonfishing'] = df['dark non-fishing'] + df['AIS non-fishing']
df['perc_dark_non_fishing'] = round((df["dark non-fishing"]/ (df["dark non-fishing"] + df["AIS non-fishing"]) * 100), 0).astype(int)
df['perc_dark_fishing'] = round((df["dark fishing"]/ (df["dark fishing"] + df["AIS fishing"]) * 100), 0).astype(int)

# %%

# %%
## How many vessels detected on average?
df.detections.sum()

# %%
## How many fishing vessels detected on average?
df.tot_fishing.sum()

# %%
## How many nonfishing vessels detected on average?
df.tot_nonfishing.sum()

# %%
## Fraction that are fishing vessels
df.tot_fishing.sum()/df.detections.sum()

# %%
## the low estimate of the fraction of vessels that are fishing
(df['dark fishing low'].sum() + df['AIS fishing'].sum() ) / df.detections.sum()

# %%
## the high estimate of the fraction of vessels that are fishing
(df['dark fishing high'].sum() + df['AIS fishing'].sum() ) / df.detections.sum()

# %%
## fraction of dark fishing
df['dark fishing'].sum()/df.tot_fishing.sum()

# %%
### What fraction of fishing is inside China?
df[df.eez_iso3=="CHN"].tot_fishing.sum()/df.tot_fishing.sum()

# %%
### What fraction of nonfishing is inside China?
df[df.eez_iso3=="CHN"].tot_nonfishing.sum()/df.tot_nonfishing.sum()

# %%

# %%
ais = df['AIS fishing'].sum()
upper_bound_dark = df.unmatched_fishing_high.sum()
lower_bound_dark = df.unmatched_fishing_low.sum()

upper = upper_bound_dark / (ais + upper_bound_dark)
lower = lower_bound_dark / (ais + lower_bound_dark)

print(f"Fraction of dark fishing between {lower:.2f} and {upper:.2f}")

# %%
ais = df['AIS non-fishing'].sum()
upper_bound_dark = df.unmatched_nonfishing_low.sum()
lower_bound_dark = df.unmatched_nonfishing_high.sum()

upper = upper_bound_dark / (ais + upper_bound_dark)
lower = lower_bound_dark / (ais + lower_bound_dark)

print(f"Fraction of dark non-fishing between {lower:.2f} and {upper:.2f}")

# %%

# %%
n = 20
d_fishing = df
d_fishing = d_fishing.groupby("eez_iso3").sum().reset_index().replace('', 'None')
d_fishing = d_fishing.set_index('eez_iso3')
d_fishing = d_fishing.sort_values("tot_fishing", ascending=False)
d_fishing = d_fishing.head(n)

df_non_fishing = df
df_non_fishing = df_non_fishing.groupby("eez_iso3").sum().reset_index().replace('', 'None')
df_non_fishing = df_non_fishing.set_index('eez_iso3')
df_non_fishing = df_non_fishing.sort_values("tot_nonfishing", ascending=False)
df_non_fishing = df_non_fishing.head(n)

fig1, axs = plt.subplots(ncols=2, nrows=1,figsize=(11, 10))
sns.set(style = 'whitegrid')

b1 = d_fishing[["dark fishing", "AIS fishing"]].plot(
    kind="barh", stacked=True, width=0.8, 
    color = ['steelblue', 'peru'], ax = axs[0],).invert_yaxis()


for x, y in enumerate(d_fishing['perc_dark_fishing']):
    axs[0].annotate(f'{y}%', (d_fishing[["dark fishing", "AIS fishing"]].sum(axis=1).astype(int)[x], x), ha='left', va='center', size=14, xytext=(3, 0),
    color = 'steelblue', textcoords='offset points')


b2 = df_non_fishing[["dark non-fishing", "AIS non-fishing"]].plot(ax=axs[1],
    kind="barh", stacked=True, width=0.8, 
    color = ['steelblue', 'peru']).invert_yaxis()

for x, y in enumerate(df_non_fishing['perc_dark_non_fishing']):
    axs[1].annotate(f'{y}%', (df_non_fishing[["dark non-fishing", "AIS non-fishing"]].sum(axis=1).astype(int)[x], x), ha='left', va='center', size=14, xytext=(3, 0),
    color = 'steelblue', textcoords='offset points')

axs[0].set_title("Fishing Activity by EEZ", fontsize = 16)
axs[1].set_title("Non-fishing Activity by EEZ", fontsize = 16)
for i in axs:
    i.spines['top'].set_visible(False)
    i.spines['right'].set_visible(False)
    i.spines['bottom'].set_visible(False)
    i.spines['left'].set_visible(False)
    i.set_ylabel('')
    i.grid(visible=False, axis = 'y')
    i.grid(visible=True, axis = 'x', color = 'dimgray')

axs[0].get_legend().remove()
axs[1].get_legend().remove()
axs[1].legend(labels = ['dark activity', 'publicly tracked'], frameon=False, fontsize = 15, loc = 'lower right')

# labels = ['dark fishing', 'matched fishing', 'dark non-fishing', 'matched non-fishing']
# fig1.legend([b1, b2], labels=labels,  frameon=False, fontsize = 15, loc = 'lower right', bbox_to_anchor=(0.95, .1))
# plt.suptitle("Fishing Activity by EEZ")
axs[0].tick_params(axis='both', which='major', labelsize=14)
axs[1].tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
# import matplotlib.pyplot as plt

plt.savefig('barchart_fishing_nonfishing_eez.png', bbox_inches="tight", dpi = 300)
plt.show()



# %%
fig1, axs = plt.subplots(ncols=2, nrows=3,figsize=(15, 8))
plt.subplots_adjust(hspace = .3, wspace=0.9)



plt.rcParams['figure.facecolor'] = 'white'

for e, (c, i) in enumerate(zip(['Africa', 'South America', 'Europe', 'North America', 'Asia',
       'Australia'],axs.ravel())):
    df_plot = df.loc[df['continent'] == c]


    df_plot = df_plot.groupby("country").sum()
    df_plot = df_plot.sort_values("tot_fishing", ascending=False)
    df_plot = df_plot.head(10)

    lower_error = df_plot['dark fishing']  - df_plot['dark fishing high']
    upper_error = df_plot['dark fishing low'] - df_plot['dark fishing']

    major_error =[lower_error, upper_error]
    zero_error = np.zeros((2,10))


    df_plot[["AIS fishing", "dark fishing"]].plot(ax = i,
        kind="barh", stacked=True, figsize=(6, 8), width=0.8,
        xerr=[zero_error, major_error]).invert_yaxis()

    i.legend(frameon=False)
    if e != 0:
        i.get_legend().remove()
    i.set(ylabel=None)
    i.set_title(f"{c}")
    # i.title("Fishing Detections by EEZ")

    i.grid(b=None)

    # idx = np.asarray([i for i in range(30)])
    # i.yticks(idx)
# plt.tight_layout()


# plt.savefig('sar_fishing.png', bbox_inches="tight", dpi = 300)
plt.show()

# %%
# Save dataframe for barchart in figure 1

# %%
d = df.groupby(['continent']).sum()
d[['AIS fishing','dark fishing','AIS non-fishing','dark non-fishing']]

# %%
# data for the bar chart for figure 1
d.to_csv('../../data/vessels_bycontinent_v20230217.csv',index=False)

# %%
n = len(d)
d = d.sort_values('tot_fishing', ascending=False)

# asymmetric_error =
lower_error = d['dark fishing']  - d['dark fishing high']
upper_error = d['dark fishing low'] - d['dark fishing']

major_error =[lower_error, upper_error]
zero_error = np.zeros((2,n))

d[["AIS fishing", "dark fishing"]].plot(
    kind="bar", stacked=True, figsize=(6, 8),
    yerr=[zero_error, major_error],
    layout='constrained'

)
plt.title("Fishing Detections by Continent")

# plt.errorbar(.25, 8000, xerr=[1000], fmt='')

# %%
n = len(d)
d = d.sort_values('tot_fishing', ascending=False)

# asymmetric_error =
lower_error = d['dark non-fishing']  - d['dark non-fishing low']
upper_error = d['dark non-fishing high'] - d['dark non-fishing']

major_error =[lower_error, upper_error]
zero_error = np.zeros((2,n))

d[["AIS non-fishing", "dark non-fishing"]].plot(
    kind="bar", stacked=True, figsize=(6, 8),
    yerr=[zero_error, major_error]

)
plt.title("Non-Fishing Detections by Continent")

# plt.errorbar(.25, 8000, xerr=[1000], fmt='')

# %%

# %%




