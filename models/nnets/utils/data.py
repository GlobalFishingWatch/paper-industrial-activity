"""
Generate train/test data from SAR thumbnails stored in Zarr.

"""
from collections import namedtuple

import zarr
import gcsfs
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def open_cloud_file(zfile, mode='r'):
    gcs = gcsfs.GCSFileSystem()
    store = gcsfs.GCSMap(zfile, gcs=gcs, check=False)
    return zarr.open(store, mode=mode, synchronizer=None)


def zarr_open(path="index.zarr/train/0", mode='r', info=False):
    """Return zarr group from local or GCS.

    If path contains gs://, it reads from GCS.

    Parameters
    ----------
    path : str
        Path to Zarr file containing images and labels

    Returns
    -------
    zarr.open(path)

    Notes
    -----
    Arrays are kept out of memory until invoked

    """
    if not path:
        root = ""
    elif "gs://" in path:
        root = open_cloud_file(path, mode)
        # root = zarr.open(
        #     f"simplecache::{path}",
        #     storage_options={"gs": {"cloud": True}},
        # )
    else:
        root = zarr.open(path, mode)  # laizy iterator

    if info:
        if isinstance(root, zarr.Group):
            print(f"\n{path}\n{root.tree()}")
        elif isinstance(root, zarr.Array):
            print(f"\n{path}\n{root.info}")
        else:
            print("\nzarr path is empty!")
    return root


def get_data_ntuple(path, variables=None):
    """Namedtuple with pointers to Zarr data (on disk).

    If path contains gs://, it reads from GCS.

    Parameters
    ----------
    path : str
        Path of Zarr file containing images and labels
    variables : list of strings
        Name of each variable (zarr array) to load

    Returns
    -------
    namedtuple with np.arrays

    Notes
    -----
    Arrays are kept out of memory until invoked
    """
    if "gs://" in path:
        root = zarr.open(
            f"simplecache::{path}",
            storage_options={"gs": {"anon": True}},
        )
    else:
        root = zarr.open(path, "r")  # laizy iterator

    print("Data:", path, root.tree())

    if not variables:
        variables = [v for v in root.keys()]

    Data = namedtuple("Data", variables)
    return Data(*[root[v] for v in variables])


def group_split(x, groups, test_size=0.2, random_state=0):
    x = np.array(x)
    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    ii, jj = list(gss.split(x, groups=groups))[0]
    return x[ii], x[jj]


def group_kfold(x, groups, n_splits=5):
    x = np.array(x)
    gkf = GroupKFold(n_splits=n_splits)
    index_pairs = list(gkf.split(x, groups=groups))
    return [(x[ii], x[jj]) for ii, jj in index_pairs]
