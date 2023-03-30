"""Two ways to process an AOI

1. Internal loop

    Pass region and define tile_dx and tile_dy
    -> single param file

2. External loop

    Pass tiles with tile_range(x1, x2, y1, y2, dx, dy)
    and comment out tile_dx and tile_dy (default to None)
    -> multi param files

(2) is better when the AOI is large, having multiple param files.

"""
from core.detector import DetectorInfraShore
from core.ranges import tile_range
from core.utils import get_min_max

folder = "output"

# AOI: Gulf of Mexico
# x1, x2, y1, y2 = get_min_max(
#     [
#         [-98.10028986693634, 31.507909404766245],
#         [-98.10028986693634, 25.408119612413582],
#         [-84.23554377318634, 25.408119612413582],
#         [-84.23554377318634, 31.507909404766245],
#     ]
# )

# AOI: North Sea
# x1, x2, y1, y2 = get_min_max(
#     [
#         [0.47211700659262945, 54.63202231864467],
#         [0.47211700659262945, 50.892099080256564],
#         [7.6352029440926295, 50.892099080256564],
#         [7.6352029440926295, 54.63202231864467],
#     ]
# )

# AOI: Scan the whole world
# x1, x2, y1, y2 = (-180, 180, -90, 90)


# ID for AOI
region_id = "NS"
suffix = "_northsea"

# GCS: gs://{bucket}/{version}/{subbucket}/{YYYYMMDD}
bucket = "scratch_fernando"
version = "v1"
subbucket = "detections"

# Mask to clip the region (AOI)
shore_buffer = "users/fernando/gulf_of_mexico_buffer_polygon_500m"

# Define time window for composite
date1, date2 = ("2021-01-01", "2021-07-01")


# region = [
#     [-92.26058992599268, 29.9234189697212],
#     [-92.26058992599268, 29.36484273346964],
#     [-91.45309480880518, 29.36484273346964],
#     [-91.45309480880518, 29.9234189697212],
# ]

# AOI: Los Angeles
# region = [
#         [-120.44689428573732, 35.363504902130295],
#         [-120.44689428573732, 32.899406341653076],
#         [-117.09606420761232, 32.899406341653076],
#         [-117.09606420761232, 35.363504902130295],
#     ]

# AOI: North Sea
region = [
         [0.47211700659262945, 54.63202231864467],
         [0.47211700659262945, 50.892099080256564],
         [7.6352029440926295, 50.892099080256564],
         [7.6352029440926295, 54.63202231864467],
     ]

# Tiles are further divide into subtiles internaly
# for k, reg in enumerate(tile_range(x1, x2, y1, y2, 20, 20)):
# for k, reg in enumerate(tile_range(x1, x2, y1, y2, 1, 1)):
DetectorInfraShore(
    region=region,  # use list for now
    # region=reg,
    bucket=bucket,
    version=version,
    subbucket=subbucket,
    # region_id=f"{region_id}_{k:02}",
    region_id=f"{region_id}_FULL",
    satellite="S1AB",
    orbit="AD",
    suffix=suffix,
    thresholdx=25,
    resolution=20,
    window_inner=140,
    window_outer=200,
    window_date=[date1, date2],
    dialation_radius=45,
    max_num_images=40,
    min_num_images=5,
    ocean_vector_uri=shore_buffer,
    tile_dx=1,
    tile_dy=1,
).process(folder=folder, skip_done=False)
