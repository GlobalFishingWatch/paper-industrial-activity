"""
Evaluate inference from the command line

Example:
    # Run on local machine

    cd ~/dev/sentinel-1-ee

    python -m classification.vessels.evaluate \
        --config-path ~/dev/sentinel-1-ee/outputs/2021-10-29/17-21-09 \
        data=gs://scratch_fernando/detection_tiles_v3.zarr \
        split=test \
        inference=/Users/fspaolo/dev/sentinel-1-ee/outputs/2021-10-29/17-21-09/inference.csv \

Note:
    Results are plotted in local dir. Move them to the "figures" dir.

Relabel/Retrain:
    infer and eval must be ran on the same split (base or null)

    You want to look at 4 values of your preferred metric
    (let’s say F1 for now), m = model, H = test set:

    - F1(m0(H0)) vs F1(m1(H0))
      How did the adding new data affect our existing holdout set
      (probably minimally)

    - F1(m0(H1)) vs  F1(m1(H1))
      How did adding new data affect new  holdout set
      (should be more)

"""
import os
import hydra

import numpy as np
import pandas as pd
import proplot as pplt
import matplotlib.pyplot as plt

from textwrap import dedent
from omegaconf import OmegaConf

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import r2_score, f1_score, mean_squared_error

from ..utils.data import zarr_open


def pull_data(data, indices, use_weight=False):
    presence = data.presence.vindex[indices].astype('f4')
    length_m = data.length_m.vindex[indices].astype('f4')
    detect_id = data.detect_id.vindex[indices]

    if use_weight and hasattr(data, 'weight'):
        weight = data.weight.vindex[indices].astype('f4')
        print('Using weights')
    else:
        weight = np.ones_like(length_m)

    return presence, length_m, detect_id, weight


def get_residuals(y_true, y_pred, fit=False):
    """Residuals w.r.t. the one-to-one line or best fit."""
    if fit:
        y_ = y_pred
    else:
        y_ = y_true
    coef = np.polyfit(y_true, y_, 1)
    y_fit = np.polyval(coef, y_true)
    y_res = y_pred - y_fit
    return y_fit, y_res


def evaluate(pred, data, index, cfg, label=""):
    """Evaluate predictions from inference code.

    pred : DataFrame
        Predictions from a CSV file.
    data : Zarr obj
        Data stored in Zarr file.
    index : Array of int or Zarr obj
        Indices into data stored in Zarr file.
        Must be the same indices used for predictions.

    """
    print("Pulling data from GCS ...")
    index = index[:]  # -> to memory

    # 2. Get ground-truth data
    presence_true, length_true, detect_id, weight = pull_data(
        data, index, use_weight=cfg.weight
    )

    # 3. Get predictions (continuous -> binary)
    presence_pred = (pred.presence.values >= 0.5).astype(int)
    length_pred = np.around(pred.length_m.values, 2)

    classes = ["noise", "vessel"]

    # 4. Use scikit-learn to get statistics
    report = classification_report(
        presence_true, presence_pred, target_names=classes
    )

    # Remove invalid lengths (presence = 0)
    mask = length_true != 0
    length_true_ = length_true[mask]
    length_pred_ = length_pred[mask]
    presence_pred_ = presence_pred[mask]
    detect_id_ = detect_id[mask]
    weight_ = weight[mask]
    index_ = index[mask]

    # prec = precision_score(presence_true, presence_pred)
    accu = accuracy_score(presence_true, presence_pred)
    f1 = f1_score(presence_true, presence_pred)
    r2 = r2_score(length_true_, length_pred_)

    rmse = mean_squared_error(
        length_true_, length_pred_, squared=False, sample_weight=weight_
    )

    print("test samples:", len(index))
    print("valid lengths:", len(index_))
    print("Classification:\n", report)

    length_fit, length_res = get_residuals(length_true_, length_pred_)
    std = np.nanstd(length_res)
    lmax = np.nanmax(length_true)

    (idx,) = np.where(np.abs(length_res) > 3 * std)

    # Outliers
    ids = detect_id_[idx]
    lpred = length_pred_[idx]
    ltrue = length_true_[idx]
    ppred = presence_pred_[idx]

    for a, b, c, d in zip(ids, lpred, ltrue, ppred):
        print(a, int(b), int(c))

    # ----- Plot ----- #

    legend = dedent(
        f"""
        presence:
        accuracy {accu*100:.1f}%
        f1-score {f1:.2f}

        length:
        r2-score {r2:.2f}
        rmse {rmse:.2f}
        """
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    ax1.plot(length_true_, length_pred_, ".")
    ax1.plot([0, lmax], [0, lmax], "-", color="0.2", linewidth=1)
    ax1.text(
        0.03,
        0.82,
        legend,
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax1.transAxes,
        fontsize=11,
    )
    ax1.set_title(label, fontsize=11)
    ax1.set_xlabel("labeled length (m)", fontsize=11)
    ax1.set_ylabel("infered length (m)", fontsize=11)

    # ss = [f"{l1:.0f}, {l2:.0f}" for l1, l2 in zip(lpred, ltrue)]
    # [ax1.text(x, y, s) for x, y, s in zip(ltrue, lpred, ss)]

    ax2.plot(length_true_, length_res, ".")
    ax2.plot(length_true_[idx], length_res[idx], ".r")
    ax2.hlines(
        [-std * 3, -std * 2, -std, std, std * 2, std * 3],
        0,
        lmax,
        colors="0.5",
        linewidth=1,
    )
    ax2.set_title(f"Residuals std = {std:.1f}, Lines: 1,2,3 stds", fontsize=11)

    ax3.hist(length_res, bins=50)
    ax3.set_title(f"Residuals std = {std:.1f}", fontsize=11)

    plt.tight_layout()

    name = cfg.inference.replace('/', '_')

    plt.savefig(f'evaluation_{name}.png', bbox_inches="tight")

    if 1:
        # Plot outliers
        for k, l_true, l_pred in zip(index_[idx], ltrue, lpred):
            tile = data.tiles[k].astype('f4')
            plt.matshow(tile[:, :, 0], cmap="bone")
            plt.title(f'label={l_true:.1f}m  pred={l_pred:.1f}m')
            plt.savefig(f'tile_{k}_{name}.png', bbox_inches="tight")

    plt.show()


@hydra.main(config_path="config", config_name="config")
def main(cfg):

    # Change run dir to src dir
    os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))

    label = (
        f"{cfg.name}: "
        f"groups={cfg.groups} "
        f"filters={cfg.base_filters} "
        f"epochs={cfg.epochs} "
        f"augment={cfg.augment} "
        f"weight={cfg.weight}"
    )

    pred = pd.read_csv(cfg.inference)
    data = zarr_open(cfg.data)
    index = zarr_open(cfg.index)

    evaluate(pred, data, index, cfg, label)


if __name__ == "__main__":
    main()
