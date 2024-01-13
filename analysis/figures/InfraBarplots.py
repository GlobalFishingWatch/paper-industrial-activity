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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

# %%
oil_bplot = pd.read_csv('../data/infra_barplot_oil_v20230816.csv')

# %%
wind_bplot = pd.read_csv('../data/infra_barplot_wind_v20230816.csv')

# %%
oil_bplot = oil_bplot[:15]
wind_bplot = wind_bplot[:6]

# %%
oil_bplot.replace('United Arab Emirates', 'Emirates', inplace=True)
wind_bplot.replace('United Arab Emirates', 'Emirates', inplace=True)

# %%
fig1, axs = plt.subplots(ncols=1, nrows=2, figsize=(5,12), gridspec_kw={'height_ratios': [2, 1]})
sns.set(style = 'whitegrid')

b1 = oil_bplot.set_index('country_name').plot(
    kind="barh", stacked=True, width=0.7, 
    color = ['#003f5c', '#ffa600'], ax = axs[0]).invert_yaxis()


for x, y in enumerate(oil_bplot.sum(axis=1).astype(int)):
    axs[0].annotate(y, (y, x), ha='left', va='center', size=14, xytext=(3, 0),
    color = '#003f5c', textcoords='offset points')


b2 = wind_bplot.set_index('country_name').plot(ax=axs[1],
    kind="barh", stacked=True, width=0.7, 
    color = ['#003f5c', '#ffa600']).invert_yaxis()

for x, y in enumerate(wind_bplot.sum(axis=1).astype(int)):
    axs[1].annotate(y, (y, x), ha='left', va='center', size=14, xytext=(3, 0),
    color = '#003f5c', textcoords='offset points')

axs[0].set_title("Oil Infrastructure 2021", fontsize = 17)
axs[1].set_title("Wind Infrastructure 2021", fontsize = 17)
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
axs[0].legend(labels = ['oil ', 'probable oil'], frameon=False, fontsize = 16, loc = 'lower right', bbox_to_anchor=(.94, 0))
axs[1].legend(labels = ['wind ', 'probable wind'], frameon=False, fontsize = 16, loc = 'lower right', bbox_to_anchor=(1.006, 0))


axs[0].tick_params(axis='both', which='major', labelsize=15)
axs[1].tick_params(axis='both', which='major', labelsize=15)

plt.savefig('barchart_oil_wind_eez.jpeg', bbox_inches="tight", dpi = 300)
plt.show()

