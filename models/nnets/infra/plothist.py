"""
plot train_loss, val_loss, val_accuracy, length_r2 vs epoch

"""
import sys

import numpy as np
import pandas as pd
# import proplot as pplot
import matplotlib.pyplot as plt

csvfile = sys.argv[1]

df = pd.read_csv(csvfile)
df = df.sort_values(by=["epoch"])
print(df.columns)

# df["val_loss"] = 1 - df.val_loss

df = df.rolling(11, center=True).mean()

fig, axes = plt.subplots(nrows=3, ncols=1)

# df.plot(x="epoch", y="loss", ax=axes[0], logy=True)
# df.plot(x="epoch", y="val_categorical_accuracy", ax=axes[0], logy=True)
# df.plot(x="epoch", y="val_loss", ax=axes[1], logy=True)
# df.plot(x="epoch", y="lr", ax=axes[2], logy=False)
df.plot(x='epoch', logy=True)

plt.show()
