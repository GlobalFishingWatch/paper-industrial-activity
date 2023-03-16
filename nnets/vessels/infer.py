"""
Run inference from the command line (flat zarr)

Examples
--------

Make inference on test set:

python -m classification.vessels.infer \
    --config-path ~/dev/sentinel-1-ee/outputs/2021-12-15/22-35-38 \
    data=gs://scratch_fernando/detection_tiles_v4.zarr \
    split=test/null \

inference -> original/run_dir/inference.csv

Notes
-----

- config path must be absolute or start with '~'
- data paths can be relative.
- To run inference on CPU with weights trained on GPU:

    CUDA_VISIBLE_DEVICES="" python your_keras_code.py

- To run inference on local machine need to download weights.
- infer and eval must be ran on the same split (base or null)

"""
import os
import logging
from datetime import datetime

import hydra
from omegaconf import OmegaConf

import numpy as np
import pandas as pd

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

from ..utils.data import zarr_open
from ..utils.cmds import download_cmd, evaluate_cmd

from .build import get_model
from .dataload import DataLoader

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def infer(model, tiles, detect_id, index, cfg):
    """Make inference and evaluate results."""
    index = index[:]  # -> to memory
    batch_size = cfg.batch_size

    logging.info("generating batches of tiles")

    # Pad tiles to make it multiple of batch size
    n_samples = len(index)
    pad = batch_size - n_samples % batch_size
    index = np.append(index, [index[-1]] * pad)

    data_loader = DataLoader(tiles, index)  # load in batches

    assert data_loader.samples % batch_size == 0

    logging.info("making predictions")

    y_pred = model.predict(data_loader)

    # Remove padding (nested list)
    y_pred[0] = y_pred[0][:-pad]  # presence
    y_pred[1] = y_pred[1][:-pad]  # length
    index = index[:-pad]

    assert len(y_pred[0]) == n_samples

    out = {
        "index": index,
        "detect_id": detect_id.vindex[index],  # out-of-mem
        "presence": np.around(y_pred[0].squeeze(), 2),
        "length_m": np.around(y_pred[1].squeeze(), 2),
    }
    pd.DataFrame(out).to_csv(cfg.inference, index=False)
    logging.info(f"inference -> {cfg.inference}")

    return cfg


@hydra.main(config_path="config", config_name="config")
def main(cfg):

    # Change run dir to src dir
    os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))

    logging.info("pulling data and index from GCS")

    data = zarr_open(cfg.data)

    if cfg.split == 'all':
        index = range(data.tiles.shape[0])
    else:
        index = zarr_open(cfg.index)

    logging.info(f"building model, infering {len(index)} samples")

    model = get_model(cfg.name, params=cfg)
    cfg = infer(model, data.tiles, data.detect_id, index, cfg)

    download_cmd(cfg.run_dir)
    evaluate_cmd(cfg.run_dir, cfg.data, cfg.split)


if __name__ == "__main__":
    now = datetime.now()
    main()
    logging.info(f"total run time: {datetime.now()-now}")
