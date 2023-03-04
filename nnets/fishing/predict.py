"""
Example:
    python predict.py

Machine:
    gcloud compute ssh fernandop-gpu-east-v100 --zone us-east1-c \
            -- -L 8265:localhost:8265

Download model from GCS:
    EVEN:

    mkdir -p /home/fspaolo/dev/sentinel-1-ee/outputs/2022-09-22/23-16-19

    gsutil -m cp -r 'gs://scratch_fernando/dev/sentinel-1-ee/outputs/2022-09-22/23-16-19/*' \
        /home/fspaolo/dev/sentinel-1-ee/outputs/2022-09-22/23-16-19/

    ODD:

    mkdir -p /home/fspaolo/dev/sentinel-1-ee/outputs/2022-09-24/22-41-57

    gsutil -m cp -r 'gs://scratch_fernando/dev/sentinel-1-ee/outputs/2022-09-24/22-41-57*' \
        /home/fspaolo//dev/sentinel-1-ee/outputs/2022-09-24/22-41-57

Upload predictions to BQ from GCS:
    bq load \
        --source_format=NEWLINE_DELIMITED_JSON \
        scratch_fernando.fishing_pred_odd_v5 \
        'gs://scratch_fernando/fishing_pred_odd_v5/*' \
        fishing:FLOAT,nonfishing:FLOAT,detect_id:STRING

"""

import os
import sys
import time
import uuid
import itertools
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List

import gcsfs
import numpy as np
import pandas as pd
import ray
import tifffile as tif
from google.cloud import storage
from ray.util.queue import Empty, Queue

from tensorflow.keras.models import load_model

sys.path.append("..")
from utils.ranges import date_range  # noqa, isort:skip

# Type aliases
DF = pd.DataFrame
Arr = np.ndarray
GCS = gcsfs.GCSFileSystem

# ----- Edit parameters ----- #

DATA_PATH = "gs://scratch_fernando/feature_tiles_all_v1"

# EVEN lon/lat
# MODEL_PATH = "/home/fspaolo/dev/sentinel-1-ee/outputs/2022-09-22/23-16-19/model"
# SUBBUCKET = "fishing_pred_even_v5"
# LOG_SAVED = "fishing_pred_even_v5.log"

# ODD lon/lat
MODEL_PATH = "/home/fspaolo/dev/sentinel-1-ee/outputs/2022-09-24/22-41-57/model"
SUBBUCKET = "fishing_pred_odd_v5"
LOG_SAVED = "fishing_pred_odd_v5.log"

BUCKET = "scratch_fernando"
PROJECT = "project-id"
CREDENTIALS = "/home/fspaolo/.config/gcloud/application_default_credentials.json"  # noqa

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ----- Data normalization ----- #

LENGTH_DIVIDER = 100.0

CHANNEL_DIVIDERS = [
    100.0,
    1.0,
    10000.0,
    1000000.0,
    10.0,
    10.0,
    10.0,
    1.0,
    0.1,
    0.1,
    10.0,
]

# NOTE: To normalize batches -> ndim=4
CHANNEL_DIVIDERS = np.array(CHANNEL_DIVIDERS, ndmin=3, dtype="f4")


def transform_channels(img: Arr, channels: List[int] = [4, 5]) -> Arr:
    for c in channels:
        img[..., c] = np.log(1 + img[..., c])
    return img


def normalize_channels(img: Arr, dividers: Arr = CHANNEL_DIVIDERS) -> Arr:
    return img / dividers


def normalize_length(arr: Arr, divider: float = LENGTH_DIVIDER) -> Arr:
    return arr / divider

# ------------------------------ #


def get_dates(dates: str) -> List[str]:
    if isinstance(dates, str):
        if ":" in dates:
            dates = dates.split(":")
            dates = date_range(dates[0], dates[1])
        elif "[" in dates:
            dates = eval(dates)
        else:
            dates = [dates]
    return [d.replace("-", "") for d in dates]


def get_file_list(path: str) -> List[str]:
    fs = gcsfs.GCSFileSystem("project-id")
    files = [fs.ls(p) for p in fs.ls(path)]
    return list(itertools.chain(*files))


def read_log(logfile: str) -> List[str]:
    """Load full file into a list."""
    if Path(logfile).exists():
        return Path(logfile).read_text().split()
    else:
        return []


def write_log(logfile: str, text: List[str]) -> None:
    """Write one entry per row."""
    with open(logfile, "a") as f:
        f.writelines([f"{s}\n" for s in text])


@ray.remote(num_cpus=1)
class Producer:

    def __init__(
        self, queue: Queue, paths: List[str], num_threads: int = 32
    ) -> None:
        self.queue = queue
        self.paths = paths
        self.num_threads = num_threads
        self.project = PROJECT
        self.credentials = CREDENTIALS
        self.gcs = gcsfs.GCSFileSystem("project-id")
        self.data = list()

    def download(self, path: str) -> None:
        if not path.startswith('gs://'):
            path = f"gs://{path}"
        name = Path(path).with_suffix("").name
        detect_id = np.array([name[:-7]])
        length = np.array([float(name[-6:]), ])
        with self.gcs.open(path, "rb") as f:
            tile = tif.imread(f)
        assert tile.shape == (100, 100, 11)
        self.data = [tile, length, detect_id]
        return self

    def transform(self) -> None:
        tile, length, detect_id = self.data
        tile = transform_channels(tile)
        tile = normalize_channels(tile)
        length = normalize_length(length)
        self.data = [tile, length, detect_id]
        return self

    def put(self) -> None:
        self.queue.put(self.data)  # wait if full

    def process(self, path: str) -> None:
        self.download(path).transform().put()

    def run(self) -> None:
        with ThreadPoolExecutor(self.num_threads) as executor:
            executor.map(self.process, self.paths)


@ray.remote(num_gpus=1)
class Consumer:

    def __init__(self, queue: Queue, batch_size: int, rank: int) -> None:
        self.queue = queue
        self.batch_size = batch_size
        self.rank = rank
        self.model = load_model(MODEL_PATH)
        self.bucket = storage.Client().bucket(BUCKET)
        self.subbucket = SUBBUCKET
        self.credentials = CREDENTIALS
        self.project = PROJECT

    def get_batch(self) -> List[Arr]:
        # batch = [self.queue.get for _ in range(self.batch_size)]
        batch = self.queue.get_nowait_batch(self.batch_size)
        return list(map(np.array, zip(*batch)))

    def save(self, df: DF) -> None:
        fname = f'results/{self.rank}-{time.time()}.json'
        df.to_json(fname, orient='records', lines=True)
        write_log(LOG_SAVED, df.detect_id.tolist())

    def upload(self, df: DF) -> None:
        fname = f'{self.subbucket}/{self.rank}-{uuid.uuid4().hex}.json'
        js = df.to_json(orient='records', lines=True)
        blob = self.bucket.blob(fname)
        blob.upload_from_string(data=js, content_type='application/json')
        write_log(LOG_SAVED, df.detect_id.tolist())

    def predict(self):
        try:
            tiles, lengths, ids = self.get_batch()
        except Empty:
            return

        N = self.batch_size
        assert tiles.shape == (N, 100, 100, 11) and lengths.shape == (N, 1)

        y_pred = self.model.predict((tiles, lengths), batch_size=64)
        y_pred = np.around(y_pred, 6)

        # fmt: off
        df = pd.DataFrame({
            "fishing": y_pred[:, 0],
            "nonfishing": y_pred[:, 1],
            "detect_id": np.squeeze(ids),
        })
        # self.save(df)
        self.upload(df)

    def run(self):
        while True:
            self.predict()
            print(f"GPU[{self.rank}] Queue[{self.queue.size()}]")


def main() -> None:
    num_prod = 42
    num_cons = 4
    batch_size = 1024
    queue_size = 1024 * 10

    print("Reading logs ...")
    logs = read_log(LOG_SAVED)

    print("Getting files ...")
    if 0:
        files = get_file_list(DATA_PATH)
        write_log('file_list.txt', files)
    else:
        files = read_log('file_list.txt')
        num_samples = int(len(files)/2)
        files = files[:num_samples]

    print("Filtering files ...")
    get_id = lambda s: s.split('/')[-1][:-11]  # noqa
    detect_ids = [get_id(s) for s in files]
    df = pd.DataFrame({"file": files, "detect_id": detect_ids})
    df = df[~df.detect_id.isin(logs)]
    files = df.file.values

    print(f"{len(files)} files")

    shards = np.array_split(files, num_prod)
    queue = Queue(maxsize=queue_size, actor_options={"num_cpus": 1})
    workers = [Producer.remote(queue, shard) for shard in shards]
    workers += [Consumer.remote(queue, batch_size, i) for i in range(num_cons)]

    ray.get([w.run.remote() for w in workers])


if __name__ == "__main__":
    t0 = datetime.now()
    main()
    print(f"TIME {datetime.now()-t0}")
