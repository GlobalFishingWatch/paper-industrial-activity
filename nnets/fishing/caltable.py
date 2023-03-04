"""Calibrate predictions on BQ tables.

Example:
    python calibrate_table.py

"""
import pickle
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd
from pathlib import Path
from google.cloud import bigquery

DF = pd.DataFrame


# NOTE: EDIT "table_name" for a new calibration!
def get_table_from_bq(model: str, name: str = 'fishing_47') -> DF:
    q = f"""
    SELECT detect_id, fishing as {name}
    FROM `project-id.scratch_fernando.fishing_pred_{model}_v5`
    """
    bq = bigquery.Client()
    return bq.query(q).to_dataframe()


def upload_df_to_bq(table_id: str, df: DF, replace: bool = True):
    """Upload a DataFrame to a BQ table.

    table_id = "project_id.dataset.table_name"

    BQ schema/types will be inferred from dataframe.
    Use pd.Timestamp(time_str) for columns w/date type.

    """
    print("Uploading to BQ ...")
    bq = bigquery.Client()

    w = "WRITE_TRUNCATE" if replace else "WRITE_APPEND"
    config = bigquery.LoadJobConfig(write_disposition=w)

    job = bq.load_table_from_dataframe(
        df, table_id, job_config=config
    )
    job.result()  # waits for job to complete

    # check loaded table
    table = bq.get_table(table_id)
    trows = table.num_rows
    drows = len(df)
    print(f"{drows} rows loaded to table ({trows} total rows)")


def calibrate_predictions(y_pred, lookup):
    getcontext().prec = 2  # Use 2 significant figures
    return np.array([lookup[Decimal(round(y, 2))] for y in y_pred])


def main():

    project = 'project-id'
    dataset = 'proj_sentinel1_v20210924'
    models = ['even', 'odd']
    field_to_calibrate = 'fishing_47'
    lookup_path = Path("../../data/calibration")
    # years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]

    for model in models:

        lookup_table1 = lookup_path / f"{model}_v5_33.pickle"
        lookup_table2 = lookup_path / f"{model}_v5_50.pickle"
        lookup_table3 = lookup_path / f"{model}_v5_66.pickle"

        with open(lookup_table1, 'rb') as f1, \
             open(lookup_table2, 'rb') as f2, \
             open(lookup_table3, 'rb') as f3:

            lookup33 = pickle.load(f1)
            lookup50 = pickle.load(f2)
            lookup66 = pickle.load(f3)

        print(f'Downloading table {model} ...')
        df = get_table_from_bq(model, field_to_calibrate)

        df['fishing_33'] = calibrate_predictions(df[field_to_calibrate], lookup33)
        df['fishing_50'] = calibrate_predictions(df[field_to_calibrate], lookup50)
        df['fishing_66'] = calibrate_predictions(df[field_to_calibrate], lookup66)
        df = df.round(6)

        print(f'Uploading table {model} ...')
        table_id = f'{project}.{dataset}.fishing_pred_{model}_v5'
        upload_df_to_bq(table_id, df, replace=True)


if __name__ == "__main__":
    main()
