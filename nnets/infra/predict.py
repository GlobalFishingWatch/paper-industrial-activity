"""Run prediction from the command line.

Read thumbnails from GCS nested TIFFs.

1. Download a batch of thumbnails from GCS
2. Predict thumbnail batch with trained model
3. Upload prediction batch to BQ
4. Save predictions to CSV (optional)

Important:
    Edit uploading parameters below.
    If run fails, delete failed table_YYYYMMDD
      from BQ and re-run prediction on full day.
    Download model from GCS:

        gsutil -m cp -r gs://bucket/folder .

Inputs:
    model: path to run_dir with model weights
    data: path to google cloud storage with tiles

Outputs:
    BQ: project_id.dataset.table_prefix_YYYYMMDD
    CSV: outdir/table_prefix_YYYYMMDD.csv

Examples:
    python -m classification.infra.predict \
        model=/home/fspaolo/dev/sentinel-1-ee/outputs/2022-07-02/21-48-35/model \
        data=gs://scratch_fernando/test_v20210924/thumbnails \
        +table=project-id.scratch_fernando.test_v20210924_predictions_20220426 \
        batch_size=1000 \

    python -m classification.infra.predict \
        model=/home/fspaolo/dev/sentinel-1-ee/outputs/2021-10-28/18-09-22/model \
        data=gfw-production-sentinel1-detection-thumbnails/v20211222 \
        +table=project-id.scratch_fernando.JUNK_predictions_20160331 \
        +date=2021-01-25 \
        batch_size=16 \

    where

        +date=2021-01-01                            # single date
        +date=2021-01-01:2021-01-15                 # date range
        +date='[2021-01-01,2021-01-02,2021-01-03]'  # list as is

Notes:
    - GCS dataset must have a nested YYYYMMDD format
    - config path must be absolute or start with '~'
    - data paths can be relative
    - To run inference on local machine need to download weights
    - To run inference on CPU with weights trained on GPU:

        CUDA_VISIBLE_DEVICES="" python predict.py

    - Thumbnails to predict are stored on:

        gs://gfw-production-sentinel1-infra-thumbnails/v20210924/<date>
        gs://scratch_fernando/infra_tiles_all_v2/<date>

    - Download model weights:

        gcloud compute scp \
            'fspaolo@fernandop-gpu-p100-1:~/dev/sentinel-1-ee/outputs/2021-11-02/18-38-27/weights*' \
            . --zone us-central1-c

Times:
    Date: 20170-01-01
    Num samples: 34284
    Batch size: 1000
    Run time 1:38:07

    Num samples: 2781861
    Batch size: 500
    fernandop-gpu-p100-1 (half) -> TIME 2 days, 21:52:18
    fernandop-gpu-p100-2 (half) -> TIME 3 days, 21:18:37

"""
import logging
import os
from datetime import datetime
from pathlib import Path

import gcsfs
import hydra
import numpy as np
import pandas as pd
import tiffile as tif
from omegaconf import OmegaConf

from tensorflow.keras.models import load_model

from ..utils.bq import upload_df_to_bq
from ..utils.cmds import download_cmd
from ..utils.ranges import date_range

# sys.path.append(Path.cwd().parent.as_posix())

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Also edit below
CLASSES = ["wind", "oil", "other", "noise"]

# Default params if no args passed
table_prefix = "detect_comp_pred_v2_"
project_id = "project-id"
dataset = "proj_sentinel1_v20210924"
outdir = "data/predictions_infra_v2"
save_csv = False


def get_dates(dates):
    """Get dates from command-line arg."""
    if isinstance(dates, str):
        if ":" in dates:
            dates = dates.split(":")
            dates = date_range(dates[0], dates[1])
        elif "[" in dates:
            dates = eval(dates)
        else:
            dates = [dates]
    return [d.replace("-", "") for d in dates]


def predict(model, tiles1, tiles2, batch_size=32):
    """Make predictions on thumbnails.

    Args:
        model: compiled Keras model
        tiles1, tiles2: 4d arrays with thumbs

    Returns:
        output: class scores (batch, num_classes)

    """
    y_pred = model.predict((tiles1, tiles2), batch_size=batch_size)
    assert len(y_pred) == len(tiles1)
    return np.around(y_pred.squeeze(), 6)


@hydra.main(config_path="config", config_name="config")
def main(cfg) -> None:

    # os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))

    model = load_model(cfg.model)

    if hasattr(cfg, "date"):
        dates = get_dates(cfg.date)
    else:
        dates = [""]

    for date in dates:

        tiles_s1 = list()
        tiles_s2 = list()
        detect_ids = list()
        paths_noext = set()

        logging.info("pulling tile list from GCS")
        fs = gcsfs.GCSFileSystem("project-id")
        tile_list = fs.ls(f"{cfg.data}/{date}")

        # Remove double extension, and remove duplicate IDs
        [paths_noext.add(Path(f).with_suffix("").with_suffix("")) for f in tile_list]

        N = len(paths_noext)  # half the length

        # Download pair of TIFF files -> numpy tiles
        for k, path_noext in enumerate(paths_noext):

            path_s1 = path_noext.as_posix() + ".VHVV20m.tif"
            path_s2 = path_noext.as_posix() + ".RGBI10m.tif"
            detect_id = path_noext.name

            logging.info(f"THUMBS {path_s1} {path_s2}")

            # Read TIFFs
            with fs.open(path_s1, "rb") as f1, fs.open(path_s2, "rb") as f2:

                tile_s1 = tif.imread(f1)
                tile_s2 = tif.imread(f2)

                if tile_s1.shape != (100, 100, 2):
                    logging.error(f"SHAPE {path_s1} {tile_s1.shape}!")
                    continue

                if tile_s2.shape != (100, 100, 4):
                    logging.error(f"SHAPE {path_s2} {tile_s2.shape}!")
                    continue

                tiles_s1.append(tile_s1)
                tiles_s2.append(tile_s2)
                detect_ids.append(detect_id)

                logging.info(f"LEN {len(tiles_s1)}")

                # Predict and upload a batch
                if len(tiles_s1) == cfg.batch_size or k == N - 1:

                    tiles_s1 = np.array(tiles_s1, dtype="f4")
                    tiles_s2 = np.array(tiles_s2, dtype="f4")

                    y_pred = predict(model, tiles_s1, tiles_s2, cfg.batch_size)

                    df = pd.DataFrame({  # noqa
                        CLASSES[0]: y_pred[:, 0],
                        CLASSES[1]: y_pred[:, 1],
                        CLASSES[2]: y_pred[:, 2],
                        CLASSES[3]: y_pred[:, 3],
                        "detect_id": detect_ids,
                    })

                    if hasattr(cfg, "table"):
                        table_id = cfg.table
                    else:
                        table_id = f"{project_id}.{dataset}.{table_prefix}{date}"

                    upload_df_to_bq(table_id, df, replace=False)

                    if save_csv:
                        csvfile = f"{outdir}/{table_prefix}{date}.csv"
                        df.to_csv(csvfile, index=False, mode="a")

                    tiles_s1 = []
                    tiles_s2 = []
                    detect_ids = []

        logging.info(f"{N} predictions uploaded for {date}")

    download_cmd(outdir)


if __name__ == "__main__":
    now = datetime.now()
    main()
    logging.info(f"TIME {datetime.now()-now}")
