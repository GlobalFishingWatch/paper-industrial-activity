"""Run thumbnail classfication training from the command line.

Example:
    Either set classification/config/config.yaml
    or pass any params through the command-line:

    python -m classification.train \
        name=model2 \
        groups=32 \
        base_filters=48 \
        epochs=300 \
        patience=1000 \
        augment=True \
        weight=False \
        split=train/null \
        data=gs://scratch_fernando/detection_tiles_v4.zarr \

"""
import os
import logging
from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import OmegaConf

from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoard,
)

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import SGD

from ..nnet.scheduler import CosineAnnealingScheduler
from ..utils.data import zarr_open

from .build import get_model
from .datagen import DataGenerator3

#   Level | Level for Humans | Level Description
#  -------|------------------|------------------------------------
#   0     | DEBUG            | [Default] Print all messages
#   1     | INFO             | Filter out INFO messages
#   2     | WARNING          | Filter out INFO & WARNING messages
#   3     | ERROR            | Filter out all messages

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def train(model, data, train_index, val_index, cfg):
    """Run training

    Parameters
    ----------
    model : Keras model
        Built and compiled model
    data : zarr obj
        Pointers to the data arrays
    index : aray of int or zarr obj
        Pointers to train/val indices into data
    cfg : omegadict obj
        Parameters and paths as dict w/dot notation

    """
    logging.info("creating data and helper functions")

    checkpoint_cb = ModelCheckpoint(
        filepath=cfg.run_dir,
        monitor=cfg.monitor,
        save_weights_only=True,
        save_best_only=True,
        mode="max",
    )
    scheduler = CosineAnnealingScheduler(
        warmup=0,
        n0=100,
        min_lr=5e-6,
        max_lr=5e-3,
        length_scale=1.0,
        max_lr_scale=1.0,
    )
    schedule_cb = LearningRateScheduler(scheduler)
    tensorboard_cb = TensorBoard(cfg.run_dir, update_freq="epoch")
    earlystop_cb = EarlyStopping(monitor=cfg.monitor, patience=cfg.patience)
    csvlog_cb = CSVLogger(cfg.history, separator=",", append=True)

    callbacks = [
        schedule_cb,
        tensorboard_cb,
        earlystop_cb,
        csvlog_cb,
        checkpoint_cb,
    ]

    if not cfg.epochs:
        cfg.epochs = scheduler.epochs_for_cycles(cfg.cycles)

    train_generator = DataGenerator3(
        data,
        train_index,
        shuffle=cfg.shuffle,
        augment=cfg.augment,
        weight=cfg.weight,
    )
    val_generator = DataGenerator3(
        data, val_index, shuffle=False, augment=False, weight=cfg.weight,
    )

    logging.info("training")

    model.fit(
        train_generator,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks,
        epochs=cfg.epochs,
        workers=1,
        verbose=1,
    )

    model.save(cfg.model)
    model.save_weights(cfg.weights)
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


@hydra.main(config_path="config", config_name="config")
def main(cfg):

    # os.chdir(hydra.utils.get_original_cwd())

    cfg.run_dir = Path().cwd().as_posix()
    cfg.src_dir = hydra.utils.get_original_cwd()
    print(OmegaConf.to_yaml(cfg))

    logging.info("building model and opening data")

    model = get_model(cfg.name, params=cfg)
    data = zarr_open(cfg.data)
    index = zarr_open(cfg.index)

    logging.info("checking index and starting train")

    train_idx, val_idx = index.train[:], index.validation[:]
    check_overlap(train_idx, val_idx, cfg.split)

    cfg = train(model, data, train_idx, val_idx, cfg)
    save_config(cfg.run_dir, cfg)


if __name__ == "__main__":
    now = datetime.now()
    main()
    logging.info(f"total run time: {datetime.now()-now}")
