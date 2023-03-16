"""Run thumbnail classfication training from the command line.

Examples:

    Either set classification/config/config.yaml
    or pass any params through the command line.

    First copy data locally on VM:

    gsutil -m cp -r 'gs://scratch_fernando/infra_tiles_v8.zarr*' .

    Then run training:

    python -m classification.infra.train_convnext2 \
        name=convnext_multi \
        batch_size=128 \
        epochs=300 \
        augment=True \
        smooth=0.1 \
        split=train/0 \
        data=/home/fspaolo/dev/sentinel-1-ee/data/infra_tiles_v8.zarr \

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
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow_addons.optimizers import AdamW

from ..nnet.callbacks import LinearDecay  # noqa
from ..nnet.scheduler import CosineAnnealingScheduler  # noqa
from ..utils.gcs import upload_dir_to_gcs
from ..utils.data import zarr_open

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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

# GCS bucket to save run dir w/model
bucket = 'scratch_fernando'


def build_model(module, cgf):
    """Build and compile model from module."""
    mdl = module.Model(
        depths=[3, 3, 9, 3],
        use_evidence=False,
        survival_prob=1,
        )

    # schedule = CosineAnnealingScheduler(
    #     warmup=0,
    #     n0=100,
    #     min_lr=5e-6,
    #     max_lr=5e-3,
    #     length_scale=1.0,
    #     max_lr_scale=1.0,
    # )

    # TODO: Implement as LearningRateSchedule subclass?
    # https://github.com/tensorflow/tensorflow/pull/11749

    # NOTE: Check history to see what lr was used

    schedule = CosineDecayRestarts(
        5e-2,        # max_lr (x 1e-1) = 5e-3
        100,         # cycle duration
        t_mul=1.0,
        m_mul=1.0,
        alpha=5e-5,  # min_lr (x 1e-1) = 5e-6
    )

    step = Variable(0, trainable=False)
    lr = lambda: 1e-1 * schedule(step)  # noqa
    wd = lambda: 1e-4 * schedule(step)  # noqa

    opt = AdamW(learning_rate=lr, weight_decay=wd)

    mdl.compile(
        optimizer=opt,
        # loss=mdl.evidential_loss(),
        # metrics=[mdl.evidential_metric(), mdl.accuracy_metric()]
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()],
    )
    return mdl


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
    logging.info("building model")

    mdl = build_model(module, cfg)

    logging.info("getting data indices")

    index_train, index_val = index.train[:], index.val[:]
    check_overlap(index_train, index_val, cfg.split)

    logging.info("setting training params")

    checkpoint_cb = ModelCheckpoint(
        filepath=cfg.run_dir,
        monitor=cfg.monitor,
        save_weights_only=True,
        save_best_only=True,
        mode="max",
    )

    # if not cfg.epochs:
    #     cfg.epochs = scheduler.epochs_for_cycles(cfg.cycles)

    # scheduler_cb = LearningRateScheduler(scheduler)
    tensorboard_cb = TensorBoard(cfg.run_dir, update_freq="epoch")
    earlystop_cb = EarlyStopping(monitor=cfg.monitor, patience=cfg.patience)
    csvlog_cb = CSVLogger(cfg.history, separator=",", append=True)
    # evidence_cb = LinearDecay(mdl.lambda_t, cfg.epochs, 0, 0.1)

    callbacks = [
        # scheduler_cb,
        tensorboard_cb,
        earlystop_cb,
        csvlog_cb,
        checkpoint_cb,
        # evidence_cb,
    ]

    dataset_train = module.DatasetSource(
        data,
        index_train,
        shuffle=cfg.shuffle,  # FIXME
        augment=cfg.augment,
        smooth=cfg.smooth,
        batch_size=cfg.batch_size,
    ).dataset()

    dataset_val = module.DatasetSource(
        data,
        index_val,
        shuffle=False,
        augment=False,
        smooth=0,
        batch_size=cfg.batch_size,
    ).dataset()

    # ------------------------------------------- #
    # Test block to plot batch of data            #
    # NOTE: To test augmentation use index_train  #
    # and shuffle=False on both datasources       #
    # ------------------------------------------- #

    # for d1, d2 in zip(dataset_train, dataset_val):
    #     try:
    #         (X1, X2), Y = d1
    #         (X1_, X2_), Y_ = d2
    #         print(X1.shape, X2.shape, Y.shape)
    #     except Exception:
    #         X, Y = d1
    #         X1, X2 = X[:, :, :, :2], X[:, :, :, 2:]
    #         print(X.shape, Y.shape)

    #     X1, X2 = X1.numpy(), X2.numpy()
    #     y = Y[0]
    #     vh = X1[0, :, :, 0]
    #     vv = X1[0, :, :, 1]
    #     rgb = X2[0, :, :, :3]
    #     nir = X2[0, :, :, -1]

    #     X1_, X2_ = X1_.numpy(), X2_.numpy()
    #     y_ = Y_[0]
    #     vh_ = X1_[0, :, :, 0]
    #     vv_ = X1_[0, :, :, 1]
    #     rgb_ = X2_[0, :, :, :3]
    #     nir_ = X2_[0, :, :, -1]

    #     plot_tiles(vh, vv, rgb, nir, y)
    #     plot_tiles(vh_, vv_, rgb_, nir_, y_)
    #     plt.show()

    # import sys; sys.exit()  # noqa

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


def plot_tiles(vh, vv, rgb, nir, label):
    # For test only
    plt.figure(figsize=(5, 5))
    plt.subplot(221)
    plt.axis("off")
    plt.imshow(vh, cmap="bone")
    plt.subplot(222)
    plt.axis("off")
    plt.imshow(vv, cmap="bone")
    plt.subplot(223)
    plt.axis("off")
    plt.imshow(rgb.astype("uint8"))
    plt.subplot(224)
    plt.axis("off")
    plt.imshow(nir, cmap="viridis")
    plt.suptitle(f"{label}")
    plt.subplots_adjust(wspace=0.005, hspace=0.01)


@hydra.main(config_path="config", config_name="config")
def main(cfg):

    # os.chdir(hydra.utils.get_original_cwd())

    cfg.run_dir = Path().cwd().as_posix()
    cfg.src_dir = hydra.utils.get_original_cwd()

    with open_dict(cfg):
        cfg.machine = socket.gethostname()

    print(OmegaConf.to_yaml(cfg))

    logging.info("getting module and data")

    path = f"classification.infra.{cfg.name}"
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
