from google.cloud import bigquery


def table_exists(table_id):
    """Test if table exists on BQ.

    table_id = project_id.dataset.table_name

    """
    client = bigquery.Client()
    try:
        client.get_table(table_id)
        return True
    except Exception as err:
        err
        return False


def upload_df_to_bq(table_id, df, replace=True):
    """Upload a DataFrame to a BQ table.

    table_id = "project_id.dataset.table_name"

    BQ schema/types will be inferred from dataframe.
    Use pd.Timestamp(time_str) for columns w/date type.

    """
    print('Uploading to BQ ...')
    client = bigquery.Client()

    w = "WRITE_TRUNCATE" if replace else "WRITE_APPEND"
    config = bigquery.LoadJobConfig(write_disposition=w)

    client.load_table_from_dataframe(
        df, table_id, job_config=config
    ).result()  # waits for job to complete

    # check loaded table
    table = client.get_table(table_id)
    trows = table.num_rows
    drows = len(df)
    print(f"{drows} rows loaded to table ({trows} total rows)")
