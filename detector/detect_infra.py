from core.detector import DetectorInfra
from core.ranges import window_range, tile_range

folder = "window_2019_west"

# Define AOI, e.g. West Hemisphere
x1, x2, y1, y2 = (-180, 0, -90, 90)

# ID for AOI
region_id = "WEST"
suffix = "_window"

# Define time window for composite
# date1, date2 = ("2020-01-01", "2020-12-30")

for date1, date2 in window_range('2019-01-01', 12, 6):
    for k, reg in enumerate(tile_range(x1, x2, y1, y2, 20, 20)):
        DetectorInfra(
            region=reg,
            region_id=f"{region_id}_{k:02}",
            suffix=suffix,
            thresholdx=21,
            resolution=20,
            window_inner=140,
            window_outer=200,
            window_date=[date1, date2],
            dialation_radius=45,
            max_num_images=40,
            min_num_images=5,
            tile_dx=1,
            tile_dy=1,
        ).process(folder=folder)
