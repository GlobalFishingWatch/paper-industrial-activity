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
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = "Roboto"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['figure.dpi'] = 600


# %%
df = pd.read_csv('../data/vessel_barplots_v20230815.csv')

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
df.replace('United Arab Emirates', 'Emirates', inplace=True)

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

fig1, axs = plt.subplots(ncols=2, nrows=1,figsize=(11, 9))

b1 = d_fishing[["dark fishing", "AIS fishing"]].plot(
    kind="barh", stacked=True, width=0.7, 
    color = [ '#ca0020', '#2c7bb6'], ax = axs[0],zorder=3).invert_yaxis()


for x, y in enumerate(d_fishing['perc_dark_fishing']):
    axs[0].annotate(f'{y}%', (d_fishing[["dark fishing", "AIS fishing"]].sum(axis=1).astype(int)[x], x), ha='left', va='center', size=14, xytext=(3, 0),
    color = '#ca0020', textcoords='offset points')


b2 = df_non_fishing[["dark non-fishing", "AIS non-fishing"]].plot(ax=axs[1],
    kind="barh", stacked=True, width=0.7, 
    color = [ '#ca0020', '#2c7bb6'], zorder=3).invert_yaxis()

for x, y in enumerate(df_non_fishing['perc_dark_non_fishing']):
    axs[1].annotate(f'{y}%', (df_non_fishing[["dark non-fishing", "AIS non-fishing"]].sum(axis=1).astype(int)[x], x), ha='left', va='center', size=14, xytext=(3, 0),
    color = '#ca0020', textcoords='offset points')

axs[0].set_title("Fishing Activity by EEZ", fontsize = 17)
axs[1].set_title("Non-fishing Activity by EEZ", fontsize = 17)
for i in axs:
    i.spines['top'].set_visible(False)
    i.spines['right'].set_visible(False)
    i.spines['bottom'].set_visible(False)
    i.spines['left'].set_visible(False)
    i.set_ylabel('')
    i.grid(visible=False, axis = 'y')
    i.grid(visible=True, axis = 'x', zorder=0)

axs[0].get_legend().remove()
axs[1].get_legend().remove()
axs[1].legend(labels = ['not publicly tracked', 'publicly tracked'], frameon=False, fontsize = 16, loc = 'lower right', bbox_to_anchor=(1.065, 0))

axs[0].tick_params(axis='both', which='major', labelsize=15)
axs[1].tick_params(axis='both', which='major', labelsize=15)
axs[0].set_axisbelow(True)
axs[1].set_axisbelow(True)

plt.tight_layout()

# plt.savefig('barchart_fishing_nonfishing_eez.jpeg', bbox_inches="tight", dpi = 300)

plt.savefig(
    "barchart_fishing_nonfishing_eez.pdf",
    transparent=True,
    bbox_inches="tight",
    pad_inches=0,
    dpi='figure',
)

plt.show()



# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
