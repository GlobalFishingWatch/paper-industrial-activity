"""Merge the interpolated 3D fields and upload to BigQuery.

Edit params below.

"""
import pandas as pd

# >>>>>>>>>> EDIT >>>>>>>>>>

names = [
    "ais_fishing",
    "ais_nonfishing",
    "dark_fishing",
    "dark_nonfishing",
    "ais_nonfishing_100",
    "dark_nonfishing_100",
]

files = [
    "data/ais_fishing_interp_v4.feather",
    "data/ais_nonfishing_interp_v4.feather",
    "data/dark_fishing_interp_v4.feather",
    "data/dark_nonfishing_interp_v4.feather",
    "data/ais_nonfishing_100_interp_v4.feather",
    "data/dark_nonfishing_100_interp_v4.feather",
]

PLOT = True
UPLOAD = True
project_id = "project-id"
dataset = "scratch_fernando"
table_name = "detections_24_w_interp_v4"

# <<<<<<<<<< EDIT <<<<<<<<<<

dfs = [pd.read_feather(f, use_threads=True) for f in files]
dfs = [df.drop(columns=["index", "i", "j", "k"]) for df in dfs]
dfs = [df.rename(columns={"v": name}) for df, name in zip(dfs, names)]
dfs = [df.astype({"lat_index": "int", "lon_index": "int"}) for df in dfs]

cols_to_index = ["date_24", "lat_index", "lon_index"]
dfs = [df.set_index(cols_to_index) for df in dfs]
table = pd.concat([df.stack() for df in dfs], axis=0).unstack()

table = table.fillna(0)
table = table.reset_index()
print(len(table))
print(table.head(10))

if UPLOAD:
    from google.cloud import bigquery

    def upload_df_to_bq(table_id, df, replace=True):
        """Upload a DataFrame to a BQ table.

        table_id = "project_id.dataset.table_name"

        BQ schema/types will be inferred from dataframe.
        Use pd.Timestamp(time_str) for columns w/date type.

        """
        print("Uploading to BQ ...")
        bq = bigquery.Client()

        w = "WRITE_TRUNCATE" if replace else "WRITE_APPEND"
        config = bigquery.LoadJobConfig(write_disposition=w)

        bq.load_table_from_dataframe(
            df, table_id, job_config=config
        ).result()  # waits for job to complete

        # check loaded table
        table = bq.get_table(table_id)
        trows = table.num_rows
        drows = len(df)
        print(f"{drows} rows loaded to table ({trows} total rows)")

    table_id = f"{project_id}.{dataset}.{table_name}"
    upload_df_to_bq(table_id, table)
    print('Table:', table_id)

if PLOT:
    import matplotlib.pyplot as plt

    table2 = table.loc[table[table[names] > 0.0].any(axis=1)]
    # table2 = table.copy()

    x, y = table2.lon_index, table2.lat_index

    plt.figure(figsize=(18, 10))
    plt.plot(x, y, ".", markersize=1.25, markeredgewidth=0)
    plt.savefig("interpolated_pixels_nonzero.png", dpi=300)
    plt.show()

print("DONE")
