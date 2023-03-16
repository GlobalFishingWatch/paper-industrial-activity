"""
Run inference from the command line (on Zarr file)

Examples:

    Make inference on test set:

    python -m classification.infra.infer \
        --config-path ~/dev/sentinel-1-ee/outputs/2022-07-02/21-48-35 \
        data=/home/fspaolo/dev/sentinel-1-ee/data/infra_tiles_v8.zarr \
        split=test/base \

    inference output -> /run_dir/inference.csv

Notes:

    - config path must be absolute or start with '~'
    - data paths can be relative.
    - To run inference on CPU with weights trained on GPU:

        CUDA_VISIBLE_DEVICES="" python your_keras_code.py

    - To run inference on local machine need to download weights.
    - infer and eval must be ran on the same split (base or null)

"""
import logging
import os
from datetime import datetime
from importlib import import_module

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from tensorflow.keras.models import load_model

from ..utils.cmds import download_cmd, evaluate_cmd
from ..utils.data import zarr_open

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Also edit below
CLASSES = ["wind", "oil", "other", "noise"]


def infer(module, data, index, cfg):
    """Make inference and evaluate results."""

    # logging.info("building model")
    # model = build_model(module, cfg)

    logging.info("loading model")

    mdl = module.Model()
    model = load_model(
        cfg.model,
        custom_objects={
            # "MyModel": module.Model,
            "loss": mdl.evidential_loss,
            "metric": mdl.evidential_metric,
        },
    )

    index = index[:]  # test index -> to memory
    batch_size = cfg.batch_size

    logging.info("generating batches of tiles")

    # Add padding to make it multiple of batch size
    n_samples = len(index)
    pad = batch_size - n_samples % batch_size
    index = np.append(index, [index[-1]] * pad)

    assert len(index) % batch_size == 0

    dataset_ = module.DatasetSource(
        data,
        index,
        shuffle=False,
        augment=False,
        smooth=0,
        batch_size=batch_size,
    ).dataset()

    logging.info("making predictions")

    y_pred = model.predict(dataset_)

    # Remove padding
    y_pred = y_pred[:-pad]
    index = index[:-pad]

    assert len(y_pred) == n_samples

    # y_pred = np.around(y_pred, 2)

    out = {
        "index": index,
        "detect_id": data.detect_id.vindex[index],  # out-of-mem
        CLASSES[0]: y_pred[:, 0],
        CLASSES[1]: y_pred[:, 1],
        CLASSES[2]: y_pred[:, 2],
        CLASSES[3]: y_pred[:, 3],
        # CLASSES[4]: y_pred[:, 4],
    }
    pd.DataFrame(out).to_csv(cfg.inference, index=False)
    logging.info(f"inference -> {cfg.inference}")

    return cfg


@hydra.main(config_path="config", config_name="config")
def main(cfg):

    # Change run dir to src dir
    os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))

    logging.info("getting module and data")

    path = f"classification.infra.{cfg.name}"
    module = import_module(path)
    data = zarr_open(cfg.data)
    index = zarr_open(cfg.index)  # test/{base,null}

    logging.info(f"infering {len(index)} samples")

    cfg = infer(module, data, index, cfg)

    download_cmd(cfg.run_dir)
    evaluate_cmd(cfg.run_dir, cfg.data, cfg.split, "infra")


if __name__ == "__main__":
    now = datetime.now()
    main()
    logging.info(f"TIME: {datetime.now()-now}")
