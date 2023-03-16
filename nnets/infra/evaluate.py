"""
Evaluate inference from the command line

Examples
--------

* Must be ran locally (on inference.csv)
* Must use same index as in inference
* It downloads full test set into memory

python -m classification.infra.evaluate \
    --config-path ~/dev/sentinel-1-ee/outputs/2022-05-09/02-08-53 \
    data=gs://scratch_fernando/infra_tiles_v3.zarr \
    split=test/base \

Relabel/Retrain
---------------

infer and eval must be ran on the same split (base or null)

You want to look at 4 values of your preferred metric
(let’s say f1score for now), m = model, H = test set:

- f1score(m0(H0)) vs f1score(m1(H0))
  How did the adding new data affect our existing holdout set
  (probably minimally)

- f1score(m0(H1)) vs  f1score(m1(H1))
  How did adding new data affect new  holdout set
  (should be more)

Metrics
-------

- In Micro-average method, you sum up the individual true
  positives, false positives, and false negatives of the
  system for different sets and then apply them to get the
  statistics.

- In Macro-average method, you take the average of the
  precision and recall of the system on different sets.

-> Micro-average is preferable if there is a class imbalance problem.

"""
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot as pplt  # noqa
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from sklearn.metrics import (  # noqa, isort:skip
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc,
)

from ..utils.data import zarr_open  # isort:skip


CLASSES = ["wind", "oil", "other", "noise"]


def get_test_set(data, indices, classes=CLASSES):
    """GCS/Zarr -> One-hot encode on the fly."""
    detect_id = data.detect_id.vindex[indices]
    label = data.label.vindex[indices]
    indices = np.array([classes.index(lb) for lb in label])
    ohe = OneHotEncoder()
    onehot = ohe.fit_transform(indices.reshape(-1, 1)).toarray()
    return detect_id, label, onehot


def multiclass_roc_auc_score(y_true, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_true, y_pred, average=average)


def save_misclassification(
    fname, classes, y_true, y_pred, label, Y_pred, detect_id
):
    ii = np.where(y_true != y_pred)
    label_miss = label[ii]
    Y_pred_miss = np.squeeze(Y_pred[ii, :])
    detect_id_miss = detect_id[ii]

    df = pd.DataFrame(columns=["label", *classes, "detect_id"])
    df["label"] = label_miss
    df["detect_id"] = detect_id_miss

    for c, cls in enumerate(classes):
        df[cls] = Y_pred_miss[:, c]

    path = "classification/infra/evals"
    Path(path).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{path}/{fname}", index=False)
    print(f"saved -> {path}/{fname}")


def evaluate(pred, data, index, cfg, classes=CLASSES):
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

    num_classes = len(classes)

    # Get ground truth from Zarr (as one-hot)
    detect_id, label, Y_true = get_test_set(data, index)

    # Get predicted scores from DataFrame
    Y_pred = pred[classes].values

    # --- Filter predictions/labels to inspect --- #

    if 0:
        uncertainty = 1 - Y_pred.sum(axis=1)
        maxscore = Y_pred.max(axis=1)  # noqa
        sumscore = Y_pred[:, :4].sum(axis=1)  # noqa
        platform_score = Y_pred[:, 1]

        print(uncertainty.min(), uncertainty.max())

        # ii, = np.where(uncertainty < 0.1)
        # ii, = np.where(maxscore > 0.5)
        (ii,) = np.where(platform_score < 0.5)
        # ii, = np.where(sumscore > 0.5)
        # ii, = np.where(label == 'other')

        print(Y_pred.shape)
        Y_pred = Y_pred[ii, :]
        Y_true = Y_true[ii, :]
        detect_id = detect_id[ii]
        label = label[ii]
        print(Y_pred.shape)
        # import sys
        # sys.exit()

    # -------------------------------------------- #

    # one-hot (matrix) -> integer (vector)
    y_true = np.argmax(Y_true, axis=1)
    y_pred = np.argmax(Y_pred, axis=1)

    save_misclassification(
        "platform_low_score.csv", classes, y_true, y_pred, label, Y_pred, detect_id
    )

    print("Computing metrics ...")

    report = classification_report(y_true, y_pred, target_names=classes)

    # Each class: ROC curve and AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average: ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_true.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(num_classes):
    #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # # Finally average it and compute AUC
    # mean_tpr /= num_classes

    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    ######

    # Each class: precision and recall
    precision, recall, threshold, f1score = {}, {}, {}, {}
    for i in range(num_classes):
        precision[i], recall[i], threshold[i] = precision_recall_curve(
            Y_true[:, i], Y_pred[:, i]
        )
        threshold[i] = np.append(threshold[i], [1])

    # Micro-average: precision and recall
    p, r, t = precision_recall_curve(Y_true.ravel(), Y_pred.ravel())
    f1 = 2 * (p * r) / (p + r)
    precision["micro"] = p
    recall["micro"] = r
    threshold["micro"] = np.append(t, [1])
    f1score["micro"] = f1

    print(report)

    # ----- PLOT ----- #

    """ Confution Matrix """

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(2, 2, 1)

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=classes,
        normalize="pred",
        colorbar=False,
        cmap="Blues",
        ax=ax,
    )
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    ax.set_title(f"Accuracy = {accuracy:.2f}  Kappa = {kappa:.2f}")

    """ ROC curves """

    plt.subplot(2, 2, 2)

    for i, c in enumerate(classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label=f"{c} (area = {roc_auc[i]:0.2f})",
        )

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-avg ROC (area = {roc_auc['micro']:0.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=3,
    )

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive rate")
    plt.ylabel("True Positive rate")
    plt.title("ROC curve per class")
    plt.legend(loc="best")

    """ Average Metrics """

    plt.subplot(2, 2, 3)

    plt.plot(
        threshold["micro"],
        precision["micro"],
        threshold["micro"],
        recall["micro"],
        threshold["micro"],
        f1score["micro"],
        lw=2,
    )

    labels = ["precision", "recall", "f1-score"]
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Micro Average Metrics")
    plt.legend(labels, loc=8)

    """ Write Text """

    plt.subplot(2, 2, 4)

    params = str(OmegaConf.to_yaml(cfg))
    idx = params.index("split")
    params = params[:idx]

    plt.text(1, 0.55, report, ha="right", family="monospace")
    plt.text(0, 0.0, params, ha="left", family="monospace")
    plt.axis("off")

    # Save figure
    path = "classification/infra/evals"
    Path(path).mkdir(parents=True, exist_ok=True)
    idx = cfg.run_dir.index("outputs") + len("outputs/")
    plt.savefig(f'{path}/{cfg.run_dir[idx:].replace("/", ":")}.png', dpi=100)

    plt.show()


@hydra.main(config_path="config", config_name="config")
def main(cfg):

    # Change to pwd for saving stuff, and get config path
    os.chdir(hydra.utils.get_original_cwd())
    cfg.run_dir = HydraConfig.get().runtime.config_sources[1].path
    print(OmegaConf.to_yaml(cfg))

    pred = pd.read_csv(cfg.inference)
    data = zarr_open(cfg.data)
    index = zarr_open(cfg.index)

    evaluate(pred, data, index, cfg)


if __name__ == "__main__":
    main()
