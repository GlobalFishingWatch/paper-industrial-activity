"""Run prediction from the command line.

Read thumbs from GCS nested JSONs or TIFFs.

1. Download a batch of thumbnails from GCS
2. Predict thumbnail batch with trained model
3. Upload prediction batch to BQ
4. Save predictions to CSV (optional)

Edit uploading parameters below.

If run fails, delete failed table_YYYYMMDD
  from BQ and re-run prediction on full day.

Inputs
------

model: path to run_dir with model weights
data: path to google cloud storage with tiles

Outputs
-------

BQ: project_id.dataset.table_prefix_YYYYMMDD
CSV: outdir/table_prefix_YYYYMMDD.csv

Examples
--------

python -m classification.vessels.predict \
    model=/home/fspaolo/dev/sentinel-1-ee/outputs/2021-10-28/18-09-22/model \
    data=gs://scratch_fernando/test_v20210924/thumbnails \
    +table=project-id.scratch_fernando.test_v20210924_predictions_20220426 \
    batch_size=1000 \


python -m classification.vessels.predict \
    model=/home/fspaolo/dev/sentinel-1-ee/outputs/2021-10-28/18-09-22/model \
    data=gfw-production-sentinel1-detection-thumbnails/v20211222 \
    +table=project-id.scratch_fernando.JUNK_predictions_20160331 \
    +date=2016-03-31 \
    batch_size=16 \

where

    +date=2021-01-01                            # single date
    +date=2021-01-01:2021-01-15                 # date range
    +date='[2021-01-01,2021-01-02,2021-01-03]'  # list as is

Notes
-----

- GCS dataset must have a nested YYYYMMDD format
- config path must be absolute or start with '~'
- data paths can be relative
- To run inference on local machine need to download weights
- To run inference on CPU with weights trained on GPU:

    CUDA_VISIBLE_DEVICES="" python predict.py

- Thumbnails to predict are stored on:

    gs://gfw-production-sentinel1-detection-thumbnails/v20210924/<date>
    gs://scratch_fernando/test_v20210924/thumbnails/<date>

- To download model weights:

    gcloud compute scp \
        'fspaolo@fernandop-gpu-p100-1:~/dev/sentinel-1-ee/outputs/2021-11-02/18-38-27/weights*' \
        . --zone us-central1-c

Run time one full day:
28934 predictions
input format: JSON
run time: 0:46h

input format: TIFF
Run time 0:37h

"""
import json
import logging
import os
# import sys
from datetime import datetime
from pathlib import Path

import gcsfs
import hydra
import numpy as np
import pandas as pd
import tiffile as tif
from omegaconf import OmegaConf

import tensorflow as tf
from tensorflow.keras.models import load_model

from ..utils.bq import upload_df_to_bq
from ..utils.cmds import download_cmd
from ..utils.ranges import date_range

# sys.path.append(Path.cwd().parent.as_posix())


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# TODO: Move this to a structured config
table_prefix = "detect_scene_pred_"
project_id = "project-id"
dataset = "proj_sentinel1_v20210924"
outdir = "data/predictions"
save_csv = False


def squared_error(y_true, y_pred):
    """Masked squared error loss/metric for length.

    Mask loss when length=0 (presence=0 -> no vessel).
    """
    mask = tf.cast(y_true > 0, "float32")
    return mask * (y_true - y_pred) ** 2


def get_dates(dates):
    if isinstance(dates, str):
        if ":" in dates:
            dates = dates.split(":")
        elif "[" in dates:
            dates = eval(dates)
        else:
            dates = [dates, 1]
        dates = date_range(dates[0], dates[1])
    return [d.replace("-", "") for d in dates]


def predict(model, tiles, batch_size=32):
    """Make predictions on thumbnails.

    model: compiled Keras model
    tiles: 4d array with thumbs (N, 80, 80, 2)

    output: presence (N), length_m (N)

    """
    y_pred = model.predict(tiles, batch_size=batch_size)
    assert len(y_pred[0]) == len(tiles)
    return y_pred[0].squeeze(), y_pred[1].squeeze()


@hydra.main(config_path="config", config_name="config")
def main(cfg) -> None:

    # os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))

    model = load_model(cfg.model, custom_objects={"squared_error": squared_error})

    if hasattr(cfg, "date"):
        dates = get_dates(cfg.date)
    else:
        dates = [""]

    for date in dates:
        logging.info("pulling tile list from GCS")
        fs = gcsfs.GCSFileSystem("project-id")
        tile_list = fs.ls(f"{cfg.data}/{date}")

        tiles = []
        detect_id = []
        N = len(tile_list)

        # Download JSON files -> numpy tiles
        for k, fname in enumerate(tile_list):

            logging.info(f"THUMB {fname}")

            # Read JSON
            # with fs.open(fname, "r") as f:
            #     json_str = json.load(f)
            #     tile = np.asarray(json_str["array"])

            # Read TIFF
            with fs.open(fname, "rb") as f:
                tile = tif.imread(f)

                if tile.shape != (80, 80, 2):
                    print(f"{fname} {tile.shape}!")
                    continue

                tiles.append(tile)
                detect_id.append(Path(fname).stem)

                logging.info(f"LEN {len(tiles)}")

                # Predict and upload a batch
                if len(tiles) == cfg.batch_size or k == N - 1:

                    presence, length_m = predict(model, np.array(tiles), cfg.batch_size)

                    df = pd.DataFrame(
                        {
                            "presence": presence,
                            "length_m": length_m,
                            "detect_id": detect_id,
                        }
                    )

                    if hasattr(cfg, "table"):
                        table_id = cfg.table
                    else:
                        table_id = f"{project_id}.{dataset}.{table_prefix}{date}"

                    upload_df_to_bq(table_id, df, replace=False)

                    if save_csv:
                        csvfile = f"{outdir}/{table_prefix}{date}.csv"
                        df.to_csv(csvfile, index=False, mode="a")

                    tiles = []
                    detect_id = []

        logging.info(f"{N} predictions uploaded for {date}")

    download_cmd(outdir)


if __name__ == "__main__":
    now = datetime.now()
    main()
    logging.info(f"Run time {datetime.now()-now}")
