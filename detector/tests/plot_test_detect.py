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
    detect_lon,
    detect_lat,
    detect_id
FROM
    world-fishing-827.scratch_fernando.test_v20210924_detections_20220426
"""

q2 = """
SELECT
    detect_lon,
    detect_lat,
    detect_id
FROM
    world-fishing-827.proj_sentinel1_v20210924.detect_scene_raw_20220426
"""

client = bigquery.Client()
df1 = client.query(q1).result().to_dataframe()
df2 = client.query(q2).result().to_dataframe()

n_orig = len(df1)
n_auto = len(df2)

df1 = df1.sort_values(by=['detect_id'])
df2 = df2.sort_values(by=['detect_id'])

x = df1.detect_lon + df1.detect_lat
y = df2.detect_lon + df2.detect_lat

fig, ax = plot.subplots(axwidth=4.5)

ax.scatter(x, y, s=0.2)
ax.format(
    xlabel=f'original lon + lat, N={n_orig}',
    ylabel=f'automated lon + lat, N={n_auto}',
    title='Detections 2022-04-26',
)
fig.savefig('figures/detection_20220426.png', bbox_inches="tight")

plot.show()
