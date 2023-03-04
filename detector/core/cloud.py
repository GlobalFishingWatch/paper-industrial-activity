import gcsfs
import tempfile
from skimage import io
from google.cloud import bigquery
from google.cloud import storage


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


def download_from_gcs(rpath, lpath, project_id="project-id"):
    try:
        fs = gcsfs.GCSFileSystem(project_id)
        out = fs.get(rpath, lpath, recursive=True)
        print(f"{len(out)} files downloaded from GCS")
    except FileNotFoundError as err:
        print(f"File(s) not found on GCS: {err}")
    except Exception as err:
        print(f"Subbucket not found on GCS: {rpath} {err}")


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


def load_img_from_gcs(bucket, name):
    """
    Read image from URL -> numpy.

    bucket: 'scratch_name'
    name: 'detection_tiles_fortnight/20170103/S1A_IW_GRDH_1SDV_20170103T000520_20170103T000549_014660_017D89_EF86;-87.4947680;12.6963240.0

    """
    client = storage.Client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(name)
    with tempfile.NamedTemporaryFile() as f:
        _ = blob.download_to_file(f)
        return io.imread(f.name)
