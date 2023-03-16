"""Run thumbnail classfication training from the command line.

Examples:
    Either set classification/config/config.yaml
    or pass any params through the command line.

    FIRST copy data locally on VM:

    gsutil -m cp -r 'gs://scratch_fernando/feature_tiles_v6.zarr*' \
        /home/fspaolo/dev/sentinel-1-ee/data/

    SECOND run training:

    python -m classification.fishing.train \
        name=convnext_mixed \
        batch_size=64 \
        epochs=60 \
        patience=100 \
        augment=True \
        normalize=True \
        smooth=0.1 \
        drop_path=False \
        layer_scale=True \
        split=train/even \
        data=/home/fspaolo/dev/sentinel-1-ee/data/feature_tiles_v5.zarr \

        data=gs://scratch_fernando/feature_tiles_v2.zarr

Notes:
    From the ConvNeXt paper:

    “We train ConvNeXts for 300 epochs using AdamW [46] with a
    learning rate of 4e-3. There is a 20-epoch linear warmup and a
    cosine decaying schedule afterward. We use a batch size of 4096
    and a weight decay of 0.05.”

    "We use the AdamW optimizer [46], data augmentation techniques
    such as Mixup [90], Cutmix [89], RandAugment [14], Random
    Erasing [91], and regularization schemes including Stochastic
    Depth [36] and Label Smoothing [69]."

Machines:
    gcloud compute ssh fernandop-gpu-p100-1 --zone us-central1-c -- \
             -L 9999:localhost:8888 \
             -R 52698:localhost:52698

    gcloud compute ssh fernandop-gpu-p100-2 --zone us-west1-b -- \
             -L 9999:localhost:8888 \
             -R 52698:localhost:52698

"""
import logging
import os
import socket
from datetime import datetime
from importlib import import_module
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, open_dict

from tensorflow import Variable
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow_addons.optimizers import AdamW
# from tensorflow.keras.optimizers import SGD

from ..nnet.callbacks import LinearDecay  # noqa
from ..nnet.scheduler import CosineAnnealingScheduler  # noqa
from ..utils.data import zarr_open
from ..utils.gcs import upload_dir_to_gcs

from tensorflow.keras.optimizers.schedules import (  # noqa, isort:skip
    PiecewiseConstantDecay,
    CosineDecayRestarts,
)

from tensorflow.keras.callbacks import (  # isort:skip
    CSVLogger,
    EarlyStopping,
    # LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)

#  Level | Level for Humans | Level Description
# -------|------------------|------------------------------------
#  0     | DEBUG            | [Default] Print all messages
#  1     | INFO             | Filter out INFO messages
#  2     | WARNING          | Filter out INFO & WARNING messages
#  3     | ERROR            | Filter out all messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# GCS bucket to save run dir w/model
bucket = "scratch_fernando"


class MirrorEpochCallback(Callback):
    """Update epoch number for lr and wd schedulers."""

    def __init__(self, var):
        self.var = var

    def on_epoch_begin(self, epoch, logs=None):
        self.var.assign(epoch)


def build_model(module, cfg):
    """Build and compile model from module."""
    mdl = module.Model(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        survival_edges=[.9, .9, .9, .9, .9],
        drop_path=cfg.drop_path,
        layer_scale=cfg.layer_scale,
    )

    # Define epochs or steps (iterations) per cycle
    epochs_per_cycle = int(cfg.epochs/3)
    # steps_per_epoch = int(cfg.num_samples / cfg.batch_size)
    # steps_per_cycle = steps_per_epoch * epochs_per_cycle

    # TODO: Implement as LearningRateSchedule subclass?
    # https://github.com/tensorflow/tensorflow/pull/11749

    # NOTE: Check history to see what lr was used

    schedule = CosineDecayRestarts(
        5e-2 * 3,           # max_lr (x 1e-1) = 5e-3
        # steps_per_cycle,  # 1 decay cycle in steps
        epochs_per_cycle,   # 1 decay cycle in epochs
        t_mul=1.0,
        m_mul=0.5,
        alpha=5e-5,          # min_lr (x 1e-1) = 5e-6
    )

    epoch = Variable(0, trainable=False)
    lr = lambda: 1e-1 * schedule(epoch)  # noqa
    wd = lambda: 1e-4 * schedule(epoch)  # noqa

    opt = AdamW(learning_rate=lr, weight_decay=wd)
    # opt = SGD(learning_rate=lr, momentum=0.999, clipnorm=1.0)

    mdl.compile(
        optimizer=opt,
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()],
    )
    return mdl, epoch


def train(module, data, index, cfg):
    """Run training

    Parameters
    ----------
    module : Python module
        Model arch and data interface
    data : zarr obj
        Pointers to the data arrays
    index : aray of int or zarr obj
        Pointers to train/val indices into data
    cfg : omegadict obj
        Parameters and paths as dict w/dot notation

    """
    logging.info("getting data indices")

    index_train, index_val = index.train[:], index.val[:]
    check_overlap(index_train, index_val, cfg.split)

    with open_dict(cfg):
        cfg.num_samples = len(index_train)

    logging.info("building model")

    mdl, epoch = build_model(module, cfg)

    logging.info("setting training params")

    checkpoint_cb = ModelCheckpoint(
        filepath=cfg.run_dir,
        monitor=cfg.monitor,
        save_weights_only=True,
        save_best_only=True,
        mode="max",
    )
    epochincrement_cb = MirrorEpochCallback(epoch)
    tensorboard_cb = TensorBoard(cfg.run_dir, update_freq="epoch")
    earlystop_cb = EarlyStopping(monitor=cfg.monitor, patience=cfg.patience)
    csvlog_cb = CSVLogger(cfg.history, separator=",", append=True)
    # evidence_cb = LinearDecay(mdl.lambda_t, cfg.epochs, 0, 0.1)

    callbacks = [
        epochincrement_cb,
        tensorboard_cb,
        earlystop_cb,
        csvlog_cb,
        checkpoint_cb,
        # evidence_cb,
    ]

    dataset_train = module.DatasetSource(
        data,
        index_train,
        shuffle=cfg.shuffle,
        augment=cfg.augment,
        normalize=cfg.normalize,
        smooth=cfg.smooth,
        batch_size=cfg.batch_size,
    ).dataset()

    dataset_val = module.DatasetSource(
        data,
        index_val,
        shuffle=False,
        augment=False,
        normalize=cfg.normalize,
        smooth=0,
        batch_size=cfg.batch_size,
    ).dataset()

    # ------------------------------------------- #
    # Block test to plot batch of data            #
    # NOTE: To test augmentation use index_train  #
    # and shuffle=False on both datasources       #
    # ------------------------------------------- #

    """
    dataset_train = module.DatasetSource(
        data,
        index_train,
        shuffle=False,
        augment=False,
        normalize=True,
        smooth=0,
        batch_size=cfg.batch_size,
    ).dataset()

    dataset_val = module.DatasetSource(
        data,
        index_train,
        shuffle=False,
        augment=True,
        normalize=True,
        smooth=0.1,
        batch_size=cfg.batch_size,
    ).dataset()

    for d1, d2 in zip(dataset_train, dataset_val):
        (X1, X2), Y = d1
        (X1_, X2_), Y_ = d2
        print(X1.shape, X2.shape, Y.shape)

        X1, X2 = X1.numpy(), X2.numpy()
        x1, x2 = X1[0], X2[0]
        y = Y[0]

        X1_, X2_ = X1_.numpy(), X2_.numpy()
        x1_, x2_ = X1_[0], X2_[0]
        y_ = Y_[0]

        plot_tiles(x1, x2, y)
        plot_tiles(x1_, x2_, y_)
        plt.show()

    import sys; sys.exit()  # noqa
    """

    # ------------------------------------------- #

    logging.info("training model")

    mdl.fit(
        dataset_train,
        validation_data=dataset_val,
        validation_steps=len(dataset_val),
        callbacks=callbacks,
        epochs=cfg.epochs,
        workers=1,
        verbose=1,
    )

    mdl.save(cfg.model, include_optimizer=False)
    logging.info("saved model and weights")

    return cfg


def save_config(path, conf, name="config.yaml"):
    Path(f"{path}/{name}").write_text(OmegaConf.to_yaml(conf))
    print(OmegaConf.to_yaml(conf))
    print(f"config -> {path}/{name}")


def check_overlap(train_idx, val_idx, split):
    if any(s in split for s in "01234"):
        msg = "train and validation indices overlap!"
        assert not (set(train_idx) & set(val_idx)), msg


def plot_tiles(tiles, length, label):
    plt.style.use("dark_background")
    fig, axs = plt.subplots(
        ncols=4, nrows=3, figsize=(15, 12), constrained_layout=True
    )
    axs = axs.ravel()
    for k in range(tiles.shape[-1]):
        tile = tiles[:, :, k]
        cidx = tile.shape[0] / 2
        axs[k].imshow(tile)
        axs[k].plot([cidx], [cidx], "+r")
    # plt.title(label, length)
    print(label, length)
    [ax.axis("off") for ax in axs]


@hydra.main(config_path="config", config_name="config")
def main(cfg):

    # os.chdir(hydra.utils.get_original_cwd())

    cfg.run_dir = Path().cwd().as_posix()
    cfg.src_dir = hydra.utils.get_original_cwd()

    with open_dict(cfg):
        cfg.machine = socket.gethostname()

    print(OmegaConf.to_yaml(cfg))

    logging.info("getting module and data")

    path = f"classification.fishing.{cfg.name}"
    module = import_module(path)
    data = zarr_open(cfg.data)
    index = zarr_open(cfg.index)  # train/{0,1,2,3,4}

    cfg = train(module, data, index, cfg)

    save_config(cfg.run_dir, cfg)
    upload_dir_to_gcs(cfg.run_dir, bucket)


if __name__ == "__main__":
    now = datetime.now()
    main()
    logging.info(f"TIME: {datetime.now()-now}")
