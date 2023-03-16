import proplot as plot
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery


def get_lonlat_from_id(detect_ids):
    lons = [float(d.split(";")[1]) for d in detect_ids]
    lats = [float(d.split(";")[2]) for d in detect_ids]
    return np.array(lons), np.array(lats)


# Download query results.
q1 = """
SELECT
    presence,
    length_m,
    detect_id
FROM
    world-fishing-827.scratch_fernando.test_v20210924_predictions2_20220426
"""

q2 = """
SELECT
    presence,
    length_m,
    detect_id
FROM
    world-fishing-827.proj_sentinel1_v20210924.detect_scene_pred_20220426
"""

client = bigquery.Client()
df1 = client.query(q1).result().to_dataframe()
df2 = client.query(q2).result().to_dataframe()

n_orig = len(df1)
n_auto = len(df2)

print(n_orig)
print(n_auto)

df1 = df1.drop_duplicates(subset=['detect_id'])
df2 = df2.drop_duplicates(subset=['detect_id'])

print(len(df1))
print(len(df2))

df1 = df1[df1.detect_id.isin(df2.detect_id)]

print(len(df1))
print(len(df2))

df1 = df1.sort_values(by=['detect_id'])
df2 = df2.sort_values(by=['detect_id'])

x1 = df1.presence
y1 = df2.presence
x2 = df1.length_m
y2 = df2.length_m

fig1, ax1 = plot.subplots(axwidth=4.5)
fig2, ax2 = plot.subplots(axwidth=4.5)

ax1.scatter(x1, y1, s=0.1)
ax1.format(
    xlabel=f'original presence, N={n_orig}',
    ylabel=f'automated presence, N={n_auto}',
    title='Predictions 2022-04-26',
)
fig1.savefig('figures/presence_20220426_2.png', bbox_inches="tight")

ax2.scatter(x2, y2, s=0.1)
ax2.format(
    xlabel=f'original length_m, N={n_orig}',
    ylabel=f'automated length_m, N={n_auto}',
    title='Predictions 2022-04-26',
)
fig2.savefig('figures/length_20220426_2.png', bbox_inches="tight")

plt.show()
