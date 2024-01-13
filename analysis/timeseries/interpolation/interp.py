# See README.txt

import time
import numpy as np
import pandas as pd
import sparse as sp

# >>>>>>>>>> EDIT >>>>>>>>>>

FIN = "data/gridded_detections_v4.feather"
FOUT = "data/{field}_interp_v4.feather"
NX, NY, NZ = 71998, 27792, 77
PLOT = False
SAVE = True

FIELDS = [
    "ais_fishing",
    "ais_nonfishing",
    "dark_fishing",
    "dark_nonfishing",
    "ais_nonfishing_100",
    "dark_nonfishing_100",
]

# <<<<<<<<< EDIT <<<<<<<<<<

df = pd.read_feather(FIN, use_threads=True)

dates = df.date_24.unique()
dates = sorted(dates)
date_to_index = {d: i for i, d in enumerate(dates)}
index_to_date = {v: k for k, v in date_to_index.items()}
xmin = df.lon_index.min()
ymin = df.lat_index.min()

df["rows"] = (df.lat_index - ymin).astype("int")
df["cols"] = (df.lon_index - xmin).astype("int")
df["steps"] = df.date_24.map(date_to_index)

for FIELD in FIELDS:
    df[f'{FIELD}_nan'] = df[FIELD]
    df.loc[df.overpasses == 0, f'{FIELD}_nan'] = np.nan

    # DOESNT WORK?!
    # nx, ny, nz = len(df.x.unique()), len(df.y.unique()), len(df.z.unique())
    # print(nx, ny, nz)

    print(df.head(10))
    print('Field:', FIELD)

    # Make sparse tensor
    data = df[f"{FIELD}_nan"]
    coords = [df.rows, df.cols, df.steps]
    Z = sp.COO(coords, data, shape=(NY, NX, NZ))
    print('Tensor:', Z)

    i_nans, j_nans, k_nans = np.where(np.isnan(Z))  # all pixels to interpolate
    ij_unique = {(i, j) for i, j in zip(i_nans, j_nans)}  # set -> unique series
    i_unique, j_unique = zip(*ij_unique)

    # Interpolate only series with at least one non-zero
    # the rest, just propage the zeros at the end

    start_time = time.time()
    kk = np.arange(NZ)
    points = list()

    for i, j in zip(i_unique, j_unique):

        s = Z[i, j, :].todense()

        if (s > 0).any():

            i_nan = np.isnan(s)
            i_val = ~i_nan
            k_interp = kk[i_nan]

            s_interp = np.interp(k_interp, kk[i_val], s[i_val])

            points.append({"i": i, "j": j, "k": k_interp, "v": s_interp})

    print("Interp:", time.time() - start_time, "sec")
    start_time = time.time()

    df_interp = pd.concat([pd.DataFrame(pts) for pts in points])

    print("Concat:", time.time() - start_time, "sec")
    start_time = time.time()
    # 790.188604

    df_interp.head(10)

    if SAVE:
        df_interp["date_24"] = [index_to_date[k] for k in df_interp.k]
        df_interp["lon_index"] = (df_interp.j + xmin).astype("int")
        df_interp["lat_index"] = (df_interp.i + ymin).astype("int")
        df_interp.reset_index().to_feather(FOUT.format(field=FIELD))
        print("Save:", time.time() - start_time, "sec")

    print(len(i_nans), len(ij_unique), len(df_interp.v))

    if PLOT:
        import matplotlib.pyplot as plt

        ij_nonzero = None  # NOTE: change this
        i_nonzero, j_nonzero = zip(*ij_nonzero)

        plt.figure(figsize=(18, 10))
        plt.plot(j_unique, i_unique, ".", markersize=0.25, markeredgewidth=0)
        plt.plot(j_nonzero, i_nonzero, ".r", markersize=0.25, markeredgewidth=0)
        plt.show()
        # plt.savefig('./grids_to_interpolate.png', dpi=300)

    if PLOT:

        def shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        i_nonzero_s, j_nonzero_s = shuffled_copies(
            np.array(i_nonzero), np.array(j_nonzero)
        )

        kk = np.arange(NZ)
        tt = np.array([index_to_date[k] for k in kk])

        plt.figure(figsize=(20, 150))

        for n, (i, j) in enumerate(zip(i_nonzero_s, j_nonzero_s)):
            px = Z[i, j, :].todense()

            i_nan = np.isnan(px)
            px_interp = np.interp(kk[i_nan], kk[~i_nan], px[~i_nan])

            plt.subplot(50, 4, n + 1)
            plt.plot(tt, px)
            plt.plot(tt[i_nan], px_interp, "or")

            if n == 50 * 4 - 1:
                break

        plt.show()
        # plt.savefig('./interpolated_series2.png', dpi=72)

print('DONE')
