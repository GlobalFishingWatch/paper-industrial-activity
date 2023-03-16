from core.detector import DetectorVessel
from core.ranges import date_range

# Local: Save params and log files
folder = "test_detect_vessel_20220426"

# GCS: gs://{bucket}/{version}/{subbucket}/{YYYYMMDD}
bucket = "scratch_fernando"
version = "test_v20210924"
subbucket = "detections"

# BQ: world-fishing-827/{dataset}{table_prefix}{YYYYMMDD}
# NOTE: Must enter this at upload time (something is not working)
dataset = ""
dataset_prefix = ""
table_prefix = ""

for date in date_range('2022-04-26', '2022-04-26'):
    DetectorVessel(
        date=date,
        bucket=bucket,
        version=version,
        subbucket=subbucket,
        dataset=dataset,
        dataset_prefix=dataset_prefix,
        table_prefix=table_prefix,
        suffix="",
        satellite="S1A",
        thresholdx=22,
        resolution=20,
        window_inner=200,
        window_outer=600,
        dialation_radius=60,
    ).process(folder=folder, skip_done=True)
    # ).save(folder=folder, skip_done=True)
