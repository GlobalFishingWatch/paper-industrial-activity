import re
import subprocess
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import shutil
import ee
import numpy as np
import six

from .gee import get_image, to_backscatter


def get_detect_id(lon, lat, scene_id):
    """Make detection ID padding with zeros to the right."""
    return f"{scene_id};{f'{lon:.6f}':<011};{f'{lat:.6f}':<010}"


def get_detect_ids(lons, lats, scene_ids):
    return [
        get_detect_id(x, y, s)
        for x, y, s in zip(lons, lats, scene_ids)
    ]


def list_files(path, glob="*.geojson"):
    """List files in `path` recursively."""
    return [str(p) for p in Path(path).rglob(glob)]


def to_bbox(coords):
    """List with four (x, y) pairs -> bbox namedtuple.

    coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    where (x, y) = [ll, ul, ur, lr] of a bbox.

    NOTE: Output must be lists (not ndarray)!

    """
    # assert len(coords) == 4, "Need 4 (x, y) pairs"
    BBOX = namedtuple("BBOX", "ll ul ur lr")
    return BBOX(*np.squeeze(coords).tolist())


def get_min_max(region):
    """Get xmin, xmax, ymin, ymax from geometry.

    Geometry is a list of coordinates [lon, lat].
    """
    xs = [x for x, y in region]
    ys = [y for x, y in region]
    return min(xs), max(xs), min(ys), max(ys)


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def rmdir(path):
    shutil.rmtree(path)
    print(f"Deleted {path}")


def read_log(logfile):
    if Path(logfile).exists():
        return Path(logfile).read_text().split()
    else:
        return list()


def write_log(logfile, text):
    if not isinstance(text, list):
        text = [text]
    with open(logfile, "a") as f:
        f.writelines([f"{s}\n" for s in text])


# ---------------------------------------------------------
# Utility functions from detector.py
# ---------------------------------------------------------


def get_times_from_scene(scene_id):
    """Returns start/end datetimes from the scene ID."""
    t = re.findall('[0-9]{8}T[0-9]{6}', scene_id)  # [t1, t2]
    fmt_in, fmt_out = "%Y%m%dT%H%M%S", "%Y-%m-%d %H:%M:%S"
    d1 = datetime.strptime(t[0], fmt_in).strftime(fmt_out)
    d2 = datetime.strptime(t[1], fmt_in).strftime(fmt_out)
    return d1, d2


def get_date_from_scene(scene_id, fmt=None):
    """Get the start time of an image using its scene id.

    fmt=None  -> datetime.date
    fmt='ymd' -> year, month, day
    fmt='str' -> YYYYMMDD

    """
    d = re.search('[0-9]{8}T[0-9]{6}', scene_id).group()
    date = datetime.strptime(d, "%Y%m%dT%H%M%S")

    if fmt == "ymd":
        return date.year, date.month, date.day
    elif fmt == "str":
        return d[:8]
    else:
        return date


def get_tasks():
    command = "earthengine task list"
    process = subprocess.Popen(
        command.split(" "), stdout=subprocess.PIPE
    )  # rm encoding
    out, err = process.communicate()
    msg = "An internal server error has occurred"
    if msg in str(out):
        print(msg)
        return None
    result = six.ensure_str(out).split("\n")
    return result


def get_running_ids(tasks):
    """Running task IDs.

    Takes a list of tasks that are generated by `earthengine
    task list` and returns the dictionary where the keys are
    the tasks ids of running (or ready to be processed) tasks
    and the values are the scene ids.

    """
    running_ids = {}
    for r in tasks:
        if "RUNNING" in r or "READY" in r:
            rr = r.split(" ")[0]
            running_ids[rr] = ""
    return running_ids


def print_dict(d):
    """Print formatted dictionary."""
    for k, v in d.items():
        print(k, ":", v)
    print()


def halt_until_completed(task, max_time=1000):
    """Wait until task has completed or exit if failed."""
    failed_state = [
        "FAILED",
        "CANCELLED",
        "CANCEL_REQUESTED",
        "UNSUBMITTED",
    ]
    while task.status()["state"] != "COMPLETED":
        if task.status()["state"] in failed_state:
            break


def check_task(task):
    print_dict(task.status())
    print("running task ...\n")
    halt_until_completed(task)
    print_dict(task.status())


def export_img_to_gcs(
    image,
    name="test",
    bucket="fernando-scratch",
    region=None,
    scale=1000,
    fmt="GeoTIFF",
    description=None,
):
    if region is not None and not isinstance(region, ee.geometry.Geometry):
        region = ee.Geometry.Polygon(region)

    # crs = image.projection().crs()
    # transform = image.projection().transform()
    task = ee.batch.Export.image.toCloudStorage(
        image=image.toFloat(),
        description=description,
        bucket=bucket,
        fileNamePrefix=name,
        scale=scale,
        region=region,
        # crs=crs,
        # crsTransform=transform,  # FIXME: see how to obtain this
        fileFormat=fmt,
        maxPixels=1e13,
    )
    task.start()
    return task


def export_img_to_drive(
    image,
    name="test",
    folder=None,
    region=None,
    scale=1000,
    fmt="GeoTIFF",
    description=None,
):
    task = ee.batch.Export.image.toDrive(
        image=image.toFloat(),
        description=description,
        fileNamePrefix=name,
        folder=folder,
        scale=scale,
        region=region,
        fileFormat=fmt,
    )
    task.start()
    return task


def get_bbox(point, width=500):
    """Point -> Feature -> buffer -> bbox."""
    bbox = (
        ee.Geometry.Point([ee.Number(point[0]), ee.Number(point[1])])
        .buffer(width / 2.0)
        .bounds()
    )
    return bbox


def sample_image(
    scene_id,
    points,
    width=500,
    scale=20,
    prefix="thumbnail",
    suffix="",
    bucket="gfw-sentinel-1-detections",
    export=True,
):
    """Sample thumbnails centered at points from scene."""
    img = get_image(scene_id)
    backscatter = to_backscatter(img)
    bboxes = [get_bbox(center, width) for center in points]
    uris = []
    tasks = []

    # The actual sampling happens at export with region
    for i, bbox in enumerate(bboxes):
        fname = f"{prefix}/{scene_id}/{suffix}{i}"

        task = ee.batch.Export.image.toCloudStorage(
            image=backscatter,
            description=scene_id,
            fileNamePrefix=fname,
            bucket=bucket,
            scale=scale,
            region=bbox,
            fileFormat="GeoTIFF",
            maxPixels=1e10,
        )
        if export:
            task.start()
        tasks.append(task)
        uris.append(f"gs://{bucket}/{fname}.tif")
    return uris, tasks


# NOTE: Remove Matplotlib dependency
# def plot_thumbnail(name, bucket="gfw-sentinel-1-detections"):
#     img = load_img_from_gcs(bucket, name)
#     plt.imshow(img)
#     plt.title(name)
#     plt.show()
#
#
# def plot_img(
#     img,
#     name="junk",
#     bucket="fernando-scratch",
#     scale=1000,
#     check=True,
#     vmin=None,
#     vmax=None,
#     region=None,
# ):
#     """Wraper for quick export->import->plot."""
#     task = export_img_to_gcs(img, name, scale=scale, region=region)
#     if check:
#         check_task(task)
#     img = load_img_from_gcs(bucket, name + ".tif")
#     plt.imshow(img, vmin=vmin, vmax=vmax)
