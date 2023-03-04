from core.detector import DetectorVessel
from core.ranges import date_range

folder = "params"

for date in date_range('2020-01-01', '2020-12-30'):
    DetectorVessel(
        date=date,
        satellite="S1A",
        suffix="_a",
        thresholdx=22,
        resolution=20,
        window_inner=200,
        window_outer=600,
        dialation_radius=60,
    ).process(folder=folder, skip_done=True)
    # ).save(folder=folder, skip_done=True)
