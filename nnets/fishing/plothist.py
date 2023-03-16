"""
plot train_loss, val_loss, val_accuracy, length_r2 vs epoch

"""
import sys

import numpy as np
import pandas as pd
import proplot as pplot
import matplotlib.pyplot as plt

csvfile = sys.argv[1]

df = pd.read_csv(csvfile)
df = df.sort_values(by=["epoch"])
print(df.columns)

# df["val_loss"] = 1 - df.val_loss

df = df.rolling(3, center=True).median()

# fig, axes = plt.subplots(nrows=3, ncols=1)

for c in df:
    if c == 'epoch': continue  # noqa
    df[c].plot(x='epoch', logy=False)
    plt.legend()
    print(f"{c} {np.nanmin(df[c].values):.2f} {np.nanmax(df[c].values):.2f}")

plt.show()
