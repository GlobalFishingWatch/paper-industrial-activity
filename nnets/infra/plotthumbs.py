import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot as plot

sys.path.append("..")  # isort:skip
from utils.data import zarr_open  # noqa, isort:skip

CLASSES = ["wind", "oil", "other", "noise"]
num_samples = 30

df = pd.read_csv(sys.argv[1])
root = zarr_open(sys.argv[2])

df = df.sample(n=num_samples, random_state=1)

# Select misclassified IDs to plot
detect_ids = root.detect_id[:].tolist()
indices = [detect_ids.index(did) for did in df.detect_id]

# Pull batch from cloud Zarr
ii1 = jj1 = np.arange(100)  # pixels
ii2 = jj2 = np.arange(100)
kk1 = np.arange(2)  # channels
kk2 = np.arange(4)
X1 = root.tiles_s1.oindex[(indices, ii1, jj1, kk1)]
X2 = root.tiles_s2.oindex[(indices, ii2, jj2, kk2)]
Y = root.label.vindex[indices]

""" Plot thumbnails """

for k, row in enumerate(df.itertuples()):
    preds = [getattr(row, cls) for cls in CLASSES]
    print(Y[k], row.label, row.detect_id)

    fig, axs = plot.subplots(nrows=2, ncols=2, space=0)
    axs[0].imshow(X1[k, :, :, 0], cmap="bone")
    axs[1].imshow(X1[k, :, :, 1], cmap="bone")
    axs[2].imshow(X2[k, :, :, :3])
    axs[3].imshow(X2[k, :, :, -1])
    axs.format(
        xticks="null",
        yticks="null",
        suptitle=f"label: {Y[k]}    pred: {np.around(preds, 3)}",
    )

    path = "evals/misclassified"
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path}/{row.detect_id}.png", dpi=100)


# ax.format(
#     xlabel=f'original lon + lat, num_samples={n_orig}',
#     ylabel=f'automated lon + lat, num_samples={n_auto}',
#     title='Detections 2022-04-26',
# )

plt.show()
