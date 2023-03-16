"""Calibrate predictions from the command line.

Examples:
    - Must be ran locally (on inference.csv)
    - Must use same index as in inference
    - It downloads full test set into memory
    - If calibrating single model, no need to pass 'inference.csv'

    python -m classification.fishing.calibrate_model \
        --config-path ~/dev/sentinel-1-ee/outputs/2022-09-24/22-41-57 \
        data=gs://scratch_fernando/feature_tiles_v5.zarr \
        split=test/base \

        ++inference1=/Users/fspaolo/dev/sentinel-1-ee/outputs/2022-09-22/23-16-19/inference.csv \
        ++inference2=/Users/fspaolo/dev/sentinel-1-ee/outputs/2022-09-24/22-41-57/inference.csv \

        --config-path ~/dev/sentinel-1-ee/outputs/2022-09-22/23-16-19 \
        --config-path ~/dev/sentinel-1-ee/outputs/2022-09-24/22-41-57 \

        inference1: EVEN
        inference2: ODD

Notes:
    Edit parameters below!

"""
import os
import pickle
from decimal import Decimal, getcontext

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot as pplt  # noqa
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import OneHotEncoder

from ..utils.data import zarr_open  # isort:skip

PULL_TEST_FROM_GCS = False
SAVE_LOOKUP_TABLE = False
LOOKUP_TABLE_NAME = "data/calibration/odd_v5_33.pickle"
LOCAL_TEST_SET = "data/calibration/y_true_test_v5.npy"

# FRAC_TO_DELETE = 0.105   # -> perferc 50-50 (y_true=0)
# FRAC_TO_DELETE = 0.539  # 33%  (y_true=0)
FRAC_TO_DELETE = 0.449


CLASSES = ["fishing", "nonfishing"]


def get_test_set(data, indices, classes=CLASSES):
    """GCS/Zarr -> One-hot encode on the fly."""
    detect_id = data.detect_id.vindex[indices]
    label = data.label.vindex[indices]
    indices = np.array([classes.index(lb) for lb in label])
    ohe = OneHotEncoder()
    onehot = ohe.fit_transform(indices.reshape(-1, 1)).toarray()
    return detect_id, label, onehot


def get_calibration_lookup(y_true, y_pred):
    """y_true with specific proportion of classes -> respective cal table."""

    # predicted probs, empirical probs
    prob_empir, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)

    x_interp = np.arange(0, 1, 0.01)
    y_interp = np.interp(x_interp, prob_pred, prob_empir)

    getcontext().prec = 2  # Use 2 significant figures
    return {Decimal(round(x, 2)): y for x, y in zip(x_interp, y_interp)}


def calibrate_predictions(y_pred, lookup):
    getcontext().prec = 2  # Use 2 significant figures
    return np.array([lookup[Decimal(round(y, 2))] for y in y_pred])


def plot_calibration_curve(name, y_true, y_pred):
    """Plot calibration curve for est w/o and with calibration."""

    lookup = get_calibration_lookup(y_true, y_pred)
    y_pred_cal = calibrate_predictions(y_pred, lookup)

    if SAVE_LOOKUP_TABLE:
        # Save lookup dictionary in binary format
        with open(LOOKUP_TABLE_NAME, "wb") as f:
            pickle.dump(lookup, f)
            print(f'Lookup table saved: {LOOKUP_TABLE_NAME}')

    plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    frac_of_pos, mean_pred_value = calibration_curve(y_true, y_pred, n_bins=10)
    frac_of_pos2, mean_pred_value2 = calibration_curve(y_true, y_pred_cal, n_bins=10)

    ax1.plot(mean_pred_value, frac_of_pos, "s-", label=f"{name}")
    ax1.plot(mean_pred_value2, frac_of_pos2, "s-", label=f"{name} calibrated")

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f"Calibration plot ({name})")

    ax2.hist(y_pred, range=(0, 1), bins=10, label=name, histtype="step", lw=2)
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")


def get_class_fraction(y_true):
    unique, counts = np.unique(y_true, return_counts=True)
    fraction = counts / np.sum(counts)
    return unique, fraction


def calibrate(pred, data, index, classes=CLASSES):
    """Evaluate predictions from inference code.

    pred : DataFrame
        Predictions from a CSV file.
    data : Zarr obj
        Data stored in Zarr file.
    index : Array of int or Zarr obj
        Indices into data stored in Zarr file.
        Must be the same indices used for predictions.

    """
    if index:
        print("Pulling data from GCS ...")

        index = index[:]  # -> to memory

        # Get ground truth from Zarr: id, label, onehot
        detect_id, label, Y_true = get_test_set(data, index)

        # Get predicted scores from DataFrame
        Y_pred = pred[classes].values

        # one-hot (matrix) -> integer (vector)
        y_true = Y_true[:, 0]
        y_pred = Y_pred[:, 0]

        # Save locally
        # np.save('data/calibration/y_true_test_v5.npy', y_true)
        # np.save('data/calibration/y_pred_ens_v5.npy', y_pred)
    else:
        # Pass data directly
        y_pred = pred
        y_true = data

    print("Computing metrics ...")

    plot_calibration_curve("Model", y_true, y_pred)

    plt.show()


@hydra.main(config_path="config", config_name="config")
def main(cfg):

    # Change to pwd for saving stuff, and get config path
    os.chdir(hydra.utils.get_original_cwd())
    cfg.run_dir = HydraConfig.get().runtime.config_sources[1].path
    print(OmegaConf.to_yaml(cfg))

    if "inference1" in cfg and "inference2" in cfg:
        # Evaluate an ensemble of predictions
        pred1 = pd.read_csv(cfg.inference1)
        pred2 = pd.read_csv(cfg.inference2)
        pred = pred1.copy()
        del pred["fishing"]
        del pred["nonfishing"]
        pred["fishing"] = (pred1.fishing + pred2.fishing) / 2
        pred["nonfishing"] = (pred1.nonfishing + pred2.nonfishing) / 2
    else:
        # Evaluate one set of predictions
        pred = pd.read_csv(cfg.inference)

    if PULL_TEST_FROM_GCS:
        # Pull y_true (test) data from GCS
        data = zarr_open(cfg.data)
        index = zarr_open(cfg.index)
        calibrate(pred, data, index)
    else:
        # Load data locally
        y_true = np.load(LOCAL_TEST_SET)
        y_pred = pred.fishing.values

        unique, frac = get_class_fraction(y_true)
        print(unique, frac)

        (i,) = np.where(y_true == 1)  # 0=remove fishing, 1=remove nonfishign
        num_to_delete = int(len(i) * FRAC_TO_DELETE)
        del_indices = np.random.choice(i, num_to_delete, replace=False)

        print(len(del_indices))

        y_true_new = np.delete(y_true, del_indices)
        y_pred_new = np.delete(y_pred, del_indices)

        unique, frac = get_class_fraction(y_true_new)
        print(unique, frac)

        # calibrate(y_pred, y_true, None)
        calibrate(y_pred_new, y_true_new, None)


if __name__ == "__main__":
    main()
