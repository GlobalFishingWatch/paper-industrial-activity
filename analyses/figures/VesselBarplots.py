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
#     display_name: rad
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
df = pd.read_csv('../../prj-global-sar-analysis/data/vessel_barplots_v20230815.csv')

# %%
df['AIS fishing'] = df.matched_fishing + df.matched_unknown_likelyfish
df['AIS non-fishing'] = df.matched_nonfishing + df.matched_unknown_likelynonfish
df['dark fishing'] = df.unmatched_fishing
df['dark non-fishing'] = df.unmatched_nonfishing
df['tot_fishing'] = df['dark fishing'] + df['AIS fishing']
df['tot_nonfishing'] = df['dark non-fishing'] + df['AIS non-fishing']
df['perc_dark_non_fishing'] = round((df["dark non-fishing"]/ (df["dark non-fishing"] + df["AIS non-fishing"]) * 100), 0).astype(int)
df['perc_dark_fishing'] = round((df["dark fishing"]/ (df["dark fishing"] + df["AIS fishing"]) * 100), 0).astype(int)

# %%
n = 20
d_fishing = df
d_fishing = d_fishing.groupby("country_name").sum().reset_index().replace('', 'None')
d_fishing = d_fishing.set_index('country_name')
d_fishing = d_fishing.sort_values("tot_fishing", ascending=False)
d_fishing = d_fishing.head(n)

df_non_fishing = df
df_non_fishing = df_non_fishing.groupby("country_name").sum().reset_index().replace('', 'None')
df_non_fishing = df_non_fishing.set_index('country_name')
df_non_fishing = df_non_fishing.sort_values("tot_nonfishing", ascending=False)
df_non_fishing = df_non_fishing.head(n)

fig1, axs = plt.subplots(ncols=2, nrows=1,figsize=(11, 13))
sns.set(style = 'whitegrid')

b1 = d_fishing[["dark fishing", "AIS fishing"]].plot(
    kind="barh", stacked=True, width=0.8, 
    color = [ '#FD7F0C', 'steelblue'], ax = axs[0],).invert_yaxis()


for x, y in enumerate(d_fishing['perc_dark_fishing']):
    axs[0].annotate(f'{y}%', (d_fishing[["dark fishing", "AIS fishing"]].sum(axis=1).astype(int)[x], x), ha='left', va='center', size=14, xytext=(3, 0),
    color = '#FD7F0C', textcoords='offset points')


b2 = df_non_fishing[["dark non-fishing", "AIS non-fishing"]].plot(ax=axs[1],
    kind="barh", stacked=True, width=0.8, 
    color = [ '#FD7F0C', 'steelblue']).invert_yaxis()

for x, y in enumerate(df_non_fishing['perc_dark_non_fishing']):
    axs[1].annotate(f'{y}%', (df_non_fishing[["dark non-fishing", "AIS non-fishing"]].sum(axis=1).astype(int)[x], x), ha='left', va='center', size=14, xytext=(3, 0),
    color = '#FD7F0C', textcoords='offset points')

axs[0].set_title("Fishing Activity by EEZ", fontsize = 16)
axs[1].set_title("Non-fishing Activity by EEZ", fontsize = 16)
for i in axs:
    i.spines['top'].set_visible(False)
    i.spines['right'].set_visible(False)
    i.spines['bottom'].set_visible(False)
    i.spines['left'].set_visible(False)
    i.set_ylabel('')
    i.grid(visible=False, axis = 'y')
    i.grid(visible=True, axis = 'x', color = 'gainsboro')

axs[0].get_legend().remove()
axs[1].get_legend().remove()
axs[1].legend(labels = ['dark activity', 'publicly tracked'], frameon=False, fontsize = 15, loc = 'lower right')

axs[0].tick_params(axis='both', which='major', labelsize=14)
axs[1].tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()

plt.savefig('barchart_fishing_nonfishing_eez.jpeg', bbox_inches="tight", dpi = 300)
plt.show()



# %%

# %%

# %%