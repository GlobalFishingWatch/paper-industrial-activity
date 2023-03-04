"""
Convenience functions for using google.cloud.storage

"""
import os
import tempfile
from io import BytesIO
from pathlib import Path

import gcsfs
import skimage
import tifffile
from google.cloud import storage

# See: https://googleapis.dev/python/storage/latest/index.html


def upload_dir_to_gcs(  # noqa
    dir_path: str, bucket_name: str, blob_name: str = None
) -> None:
    """Upload content of dir recursively to a blob.

    If no blob_name, recreates paths relative to home.
    """
    dir_path = Path(dir_path).resolve()
    local_paths = Path(dir_path).rglob("*")
    gcs = storage.Client()
    bucket = gcs.get_bucket(bucket_name)

    if not blob_name:
        blob_name = dir_path.relative_to(Path().home())

    for local_file in local_paths:
        remote_path = blob_name / Path(local_file).relative_to(dir_path)

        if local_file.is_file():
            print(f"Uploading {bucket_name}/{remote_path} ...")
            blob = bucket.blob(remote_path.as_posix())
            blob.upload_from_filename(local_file.as_posix())


# TODO
# def download_dir_from_gcs()


def download_from_gcs(rpath, lpath, project_id="project-id"):
    try:
        fs = gcsfs.GCSFileSystem(project_id)
        out = fs.get(rpath, lpath, recursive=True)
        print(f"{len(out)} files downloaded from GCS")
    except FileNotFoundError as err:
        print(f"File(s) not found on GCS: {err}")
    except Exception as err:
        print(f"Subbucket not found on GCS: {rpath} {err}")


def load_img_from_gcs(bucket, name):
    """
    Read image from URL -> numpy.

    bucket: 'scratch_fernando'
    name: 'detection_tiles_fortnight/20170103/S1A_IW_GRDH_1SDV_20170103T000520_20170103T000549_014660_017D89_EF86;-87.4947680;12.6963240.0

    """
    gcs = storage.Client()
    bucket = gcs.bucket(bucket)
    blob = bucket.blob(name)
    with tempfile.NamedTemporaryFile() as f:
        _ = blob.download_to_file(f)
        return skimage.io.imread(f.name)


def decompose_gcs_path(gcs_path):
    """Break a gcs path into bucket and blob names

    Parameters
    ----------
    gcs_path : str
        Format is "gs://BUCKET_NAME/PATH/TO/BLOB"

    Returns
    -------
    str
        BUCKET_NAME
    str
        PATH/TO/BLOB
    """
    gcs_path = str(gcs_path)
    assert gcs_path.startswith("gs://"), f"bad GCS path: {gcs_path}"
    gcs_path = gcs_path[5:]
    bucket_name, blob_name = gcs_path.split("/", 1)
    return bucket_name, blob_name


def gcs_path_to_blob(gcs_path, create=False):
    """Get or create a blob object based on a gcs_path.

    Parameters
    ----------
    gcs_path : str
        Format is "gs://BUCKET_NAME/PATH/TO/BLOB"
    create : bool, optional
        If true create a new blob, otherwise get an existing blob

    Returns
    -------
    google.cloud.storage.blob.Blob
    """
    bucket_name, blob_name = decompose_gcs_path(gcs_path)
    gcs = storage.Client()
    bucket = gcs.get_bucket(bucket_name)
    if create:
        return bucket.blob(blob_name)
    else:
        return bucket.get_blob(blob_name)


def list_gcs_files(gcs_path):
    """List the gcs files at the given path

    Parameters
    ----------
    gcs_path : str
        "gs://BUCKET_NAME/FILTER/PREFIX"

    Returns
    -------
    list
        Values are full GCS paths of the form "gs://BUCKET_NAME/PATH/TO/BLOB"
    """
    bucket_name, prefix = decompose_gcs_path(gcs_path)
    gcs = storage.Client()
    bucket = gcs.get_bucket(bucket_name)
    return [f"gs://{bucket.name}/{x.name}" for x in bucket.list_blobs(prefix=prefix)]


def write_text(text, path):
    """
    Parameters
    ----------
    text : str
    path : pathlib.Path or str

    Returns
    -------
    pathlib.Path or str
    """
    # FIXME: AttributeError: 'PosixPath' object has no attribute 'asposix'
    # FIXME: Removing for now
    # strpath = path.asposix() if isinstance(path, Path) else path
    strpath = str(path)
    if strpath.startswith("gs://"):
        f = BytesIO()
        f.write(text)
        gcs_path_to_blob(strpath, create=True).upload_from_file(f, rewind=True)
    else:
        # FIXME: The default (w/o args) is 'read only', added 'a'
        with open(path, "a") as f:
            f.write(text)
    return path


def load_tiff(path):
    assert os.path.splitext(path)[1].lower() in (
        ".tiff",
        ".tif",
    ), f"bad suffix on tiff path: {path}"
    if path.startswith("gs://"):
        raw = gcs_path_to_blob(path).download_as_bytes()
        return tifffile.imread(BytesIO(raw), name=os.path.basename(path))
    else:
        return tifffile.imread(path)


def read_text(path):
    """
    Parameters
    ----------
    path : pathlib.Path or str

    Returns
    -------
    str
    """
    strpath = path.asposix() if isinstance(path, Path) else path
    if strpath.startswith("gs://"):
        raw = gcs_path_to_blob(path).download_as_bytes()
        return raw.decode("utf8")
    else:
        with open(path) as f:
            return f.read()
