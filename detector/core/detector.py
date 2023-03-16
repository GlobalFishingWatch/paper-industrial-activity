"""CFAR implementation for Sentinel-1 on Google Earth Engine.

Classes:
    Params - handles input parameters, files and table names
    Detector - Applies CFAR to a generic EE image object
    DetectorVessel - Applies `Detector` to a collection of scenes
    DetectorInfra - Applies `Detector` to an image composite
    DetectorInfraShore - Applies `DetectoInfra` to onshore/land area
    ExportFootprint - Applies `DetectorVessel` to get footprints only

Modules:
    detector.py - the various `Detector` classes
    params.py - the Params class
    footprint.py - the ExportFootprint class
    gee.py - functions to interact with GEE
    ranges.py - date, tile, window range iterators
    cloud.py - gcs and bq helper functions
    utils.py - generic helper functions
    checks.py - name check helper functions

Scripts:
    detect_vessel.py - run vessel detector
    detect_infra.py - run fixed-infra detector
    detect_shore.py - run onshore detector
    export_fprnts.py - run footprint generator
    upload.py - download/process/upload detections
    match.py - match detections to AIS
    evaluate.py - assess matched detections

    upload_fprnts.py
    rasterize_fprnts.py
    interpolate_ais.py
    s1_positions.py

"""
import json
import time
from datetime import datetime
from pathlib import Path

import ee

from .gee import (
    clip_polygon,
    get_centroids,
    get_composite,
    get_connected_pixels,
    get_image,
    get_image_footprint,
    get_images,
    get_mean_donut,
    get_ocean_footprint,
    get_ocean_vector,
    get_scenes_over_ocean,
    get_water_raster,
    mask_water,
    moving_mean,
    to_backscatter,
    to_geometry,
)
from .params import Params
from .utils import get_min_max, to_bbox
from .ranges import tile_range

CONFIG = "config.yaml"

# Uncomment to set a different account
# ee.Authenticate()
# ee.Initialize()


class Detector(object):
    """Set and run CFAR detection algorithm on GEE.

    Process one GEE image object.

    To process multiple scenes or dates or a composite use:

        DetectorVessel or DetectorInfra

    To use/test different paramters, loop through params with:

        Params.combine()

    Example:

        from detector import Detector, Params

        resolutions = [20, 40]
        thresholds = [15, 20, 25]
        windows = [400, 600, 800]
        dates = ["2019-09-05", "2020-01-02", "2020-05-01"]

        for r, t, w, d in Params.combine(
            resolutions, thresholds, windows, dates
        ):
            Detector(
                resolution=r,
                thresholdx=t,
                window_outer=w,
                date=d
            ).process(check_every=60)

    """

    def __init__(self, config=CONFIG, **kwargs):
        self.params = Params(config).replace(**kwargs)

    def detect(self, img, ocean_footprint=None):
        """Detect objects on a (single or composite) image.

        Input: image object, valid geometry, and parameters from config.yaml.
        Output: detection centroids.

        Notes
        -----
        `img` should be ready to process, i.e. clipped, in backscatter,
        and tiled. These operations are performed outside 'detect()'.

        If no 'ocean_footprint', it detects over the full image.

        """
        thresholdx = self.params.thresholdx
        use_seive = self.params.use_seive
        resolution = self.params.resolution
        window_inner = self.params.window_inner
        window_outer = self.params.window_outer
        dialation_radius = self.params.dialation_radius

        if not ocean_footprint:
            ocean_footprint = img.geometry()

        """ Sea-clutter averaging strategy """

        backscatter = img

        # NOTE: use sum instead of mean (@tim)
        backscatter_mean_outer = moving_mean(
            backscatter,
            window=window_outer,
        )
        backscatter_mean_inner = moving_mean(
            backscatter,
            window=window_inner,
        )
        backscatter_mean = get_mean_donut(
            backscatter_mean_outer,
            backscatter_mean_inner,
            window_outer,
            window_inner,
        )

        # STD = sqrt(E[x^2] - E[x]^2)
        backscatter_mean2_outer = moving_mean(
            backscatter.pow(2),
            window=window_outer,
        )
        backscatter_mean2_inner = moving_mean(
            backscatter.pow(2),
            window=window_inner,
        )
        backscatter_mean2 = get_mean_donut(
            backscatter_mean2_outer,
            backscatter_mean2_inner,
            window_outer,
            window_inner,
        )
        backscatter_std = (
            backscatter_mean2.subtract(backscatter_mean.pow(2))
        ).pow(0.5)

        """ Threshold and candidate pixels """

        # tau = mean_b + n * std_b
        threshold = backscatter_mean.add(
            ee.Image(thresholdx).multiply(backscatter_std)
        )

        candidate_pixels = backscatter.select("VH").gte(
            threshold
        )  # Mask

        # Duplicate band, required for GEE function below (get_centroids)
        candidate_pixels = candidate_pixels.addBands(
            candidate_pixels.rename("VH_duplicate")
        )

        if use_seive:
            # update candidate_pixels with connected ones
            candidate_pixels = get_connected_pixels(candidate_pixels)

        # NOTE: Erode and Dialate by erosion_radius and dialation_radius.
        # Note that we are not using any erosion radius here, and the
        # dialation radius has been fixed at 60m. This elimintes all
        # single point detections.

        if dialation_radius:
            candidate_pixels_cleaned = candidate_pixels.focal_max(
                radius=dialation_radius,
                kernelType="square",
                units="meters",
            )
        else:
            candidate_pixels_cleaned = candidate_pixels

        # Calculate centroids of connected clusters
        centroids = get_centroids(
            candidate_pixels_cleaned,
            img.geometry(),
            ocean_footprint,
            resolution,
        )

        # NOTE: for some reason, the above returns a center point for
        # the scene with a label of 0. We don't know why it does this,
        # but we don't want that detection. Select only 1.

        # Convert centroids ee.FeatureCollection() into a raster
        # centroids_raster = ee.Image().byte().paint(centroids)

        # Add 2 bands to centroids_raster where values are lon/lat coords
        # centroids_lonlat_raster = centroids_raster.addBands(
        #     ee.Image.pixelLonLat()
        # )

        # connected_pixel_count = candidate_pixels.mask(
        #     candidate_pixels
        # ).connectedPixelCount(maxSize=50, eightConnected=True)

        self.centroids = centroids
        return self

    def export(self, collection=None):
        """Export detection centroids to GCS.

        This method initializes all lazy computations on GEE.
        If collection, export footprint polygons.

        For a single image:

            Detector().detect().export()

        """
        bucket = self.params.bucket
        version = self.params.version
        subbucket = self.params.subbucket
        scene_id = self.params.scene_id
        date = self.params.date.replace("-", "")

        if not collection:
            collection = ee.FeatureCollection(self.centroids)

        # File name on GCS
        fname = f"{version}/{subbucket}/{date}/{scene_id}"

        task = ee.batch.Export.table.toCloudStorage(
            collection=collection,
            description=scene_id,
            bucket=bucket,
            fileNamePrefix=fname,
            fileFormat="GeoJSON",
        )
        task.start()

        if not hasattr(self, "tasks"):
            self.tasks = list()
        if not hasattr(self, "files"):
            self.files = list()

        self.task = task
        self.tasks.append(task)
        self.files.append(fname + ".geojson")
        return self

    def check(self, start_time=None, every=60, max_time=1e6):
        """Check GEE active tasks using the Python API."""
        if every < 1:
            return self

        if not hasattr(self, 'tasks'):
            print('No EE tasks created')
            return self

        tasks = self.tasks
        n_tasks = len(tasks)

        if start_time is None:
            start_time = datetime.now()

        while 1:
            n_ready = 0
            n_running = 0
            n_completed = 0

            time.sleep(every)  # check running tasks every n secs

            minutes = (
                datetime.now() - start_time
            ).total_seconds() / 60.0

            for tsk in tasks:
                state = tsk.status()["state"]
                n_ready += 1 if state == "READY" else 0
                n_running += 1 if state == "RUNNING" else 0
                n_completed += 1 if state == "COMPLETED" else 0

            print(
                (
                    "ready: {}, running: {}, completed: {}/{}, "
                    "minutes: {}, tasks/min: {}"
                ).format(
                    n_ready,
                    n_running,
                    n_completed,
                    n_tasks,
                    int(minutes),
                    round(n_completed / float(minutes), 1),
                )
            )
            if (
                n_ready == 0 and n_running == 0
            ) or minutes > max_time:
                break

        failed_state = [
            "FAILED",
            "CANCELLED",
            "CANCEL_REQUESTED",
            "UNSUBMITTED",
        ]
        failed = {}

        for tsk in tasks:
            status = tsk.status()
            if status["state"] in failed_state:
                failed[status["description"]] = status

        self.failed = failed
        print("Tasks failed:", len(failed))
        return self

    def log(self):
        """Save run info (success and failed scenes) to a JSON file."""

        for attr in ["scenes", "failed", "files"]:
            if not hasattr(self, attr):
                setattr(self, attr, list())

        info = {
            "scenes": self.scenes,
            "failed": self.failed,
            "files": self.files,
        }

        fname = self.params.filename.replace(
            self.params.param_prefix, self.params.log_prefix
        )
        fout = Path(self.params.run_dir) / Path(fname)

        with open(fout, "w") as f:
            json.dump(info, f, indent=4)

        print("SCENES ->", fout)
        return self

    def save(self, folder=None, skip_done=None):
        # if hasattr(self, 'tasks'):
        self.params.save(folder)
        self.log()
        return self

    def print(self):
        self.params.print()
        return self

    def param_exists(self):
        return self.params._file_exists()


class DetectorVessel(Detector):
    """Run CFAR detector on single or sequence of images.

    Extends the class `Detector`.

    """

    def __init__(self, *args, **kwargs):
        """Fixup specific paramters at instantiation."""
        super().__init__(*args, **kwargs)
        assert self.params.date, "A date or scene_id must be provided!"
        self.params.replace(comp_suffix="")

    def get_scenes(self):
        """Get scenes overlapping the ocean for a date."""
        if self.params.scene_id:
            return [self.params.scene_id]

        date = self.params.date
        satellite = self.params.satellite
        ocean_vector_uri = self.params.ocean_vector_uri

        ocean_vector = get_ocean_vector(ocean_vector_uri)
        return get_scenes_over_ocean(date, ocean_vector, satellite)

    def get_footprint(self, img):
        """Get detection footprint (image intersect. water)."""
        water_raster = get_water_raster(
            self.params.ocean_raster_uri, self.params.shore_buffer
        )
        image_footprint = get_image_footprint(
            img,
            water_raster,
            scale=self.params.foot_scale,
            resolution=self.params.resolution,
            percentile=self.params.foot_percentile,
            buffer=self.params.foot_buffer,
            usevv=self.params.foot_usevv,
        )
        return get_ocean_footprint(
            img,
            image_footprint,
            water_raster,
            self.params.foot_scale,
        )

    def process(self, folder=None, skip_done=True, check_every=60):
        """Run detect().export().save() on a sequence of scenes.

        Input: one scene_id or one date.
        Output: a list of GEE tasks returned by export() to GCS.

        Notes
        -----
        In the event that both `scene_id` and `date` are defined,
        it gives priority to single `scene` processing.

        """
        self.params.replace(run_dir=folder)

        if skip_done and self.param_exists():
            print(f"Skipping {self.params.filename}, processed")
            return self

        date = self.params.date
        region = self.params.region

        scenes = self.get_scenes()

        n_scenes = len(scenes)

        if n_scenes == 0:
            print("No scenes over the ocean for:", date)
            return self
        elif n_scenes == 1:
            print(f"Processing scene {scenes[0]}")
        else:
            print(f"Processing date {date} ({n_scenes} scenes)")

        self.scenes = scenes

        for scene_id in scenes:

            self.params.scene_id = scene_id

            image = get_image(scene_id)
            footprint = self.get_footprint(image)

            image = to_backscatter(image)
            image = clip_polygon(image, footprint)
            image = clip_polygon(image, region)
            image = image.select("VH")

            self.detect(image, footprint).export()

        # One param file per day
        self.check(every=check_every).save(folder)

        return self


class DetectorInfra(Detector):
    """Run CFAR detector on a composite."""

    def __init__(self, *args, **kwargs):
        """Extend to fixup specific paramters."""
        super().__init__(*args, **kwargs)
        msg = "A date range/window must be provided!"
        assert self.params.window_date, msg
        self.params.replace(date=self.params.window_date[0])
        if not self.params.comp_suffix:
            self.params.replace(comp_suffix='_comp')

    def get_tiles(self):
        region = to_bbox(self.params.region)  # -> namedtuple
        dx = self.params.tile_dx
        dy = self.params.tile_dy

        if dx and dy:
            x1, x2, y1, y2 = get_min_max(region)
            tiles = list(tile_range(x1, x2, y1, y2, dx, dy))
        else:
            tiles = [region]

        assert len(tiles) > 0, "No tiles: Check region and tile limits"

        return [to_geometry(tile) for tile in tiles]

    def get_tile_id(self, k):
        """Generate 'tile ID' (matching scene_id)."""
        date1, date2 = self.params.window_date
        satellite = self.params.satellite
        orbit = self.params.orbit
        reducer = self.params.reducer.upper()
        region_id = self.params.region_id
        comp_suffix = self.params.comp_suffix.upper()
        return (
            f"{satellite}_{orbit}_{reducer}{comp_suffix}"
            f"_{date1.replace('-', '')}T000000"
            f"_{date2.replace('-', '')}T000000"
            f"_{region_id.upper()}_TILE_{k:03}"
        )

    def process(self, check_every=60, folder=None, skip_done=False):
        """Run detect().export() on a composite for a region.

        Input: a date range, a region, and specific params.
        Output: a list of GEE tasks returned by export() to GCS.

        """
        self.params.replace(run_dir=folder)

        if skip_done and self.param_exists():
            print(f"Skipping {self.params.filename}")
            return self

        date1, date2 = self.params.window_date
        satellite = self.params.satellite
        orbit = self.params.orbit
        shore_buffer = self.params.shore_buffer
        min_num_images = self.params.min_num_images  # per pixel
        max_num_images = self.params.max_num_images  # in collection
        ocean_vector_uri = self.params.ocean_vector_uri
        ocean_raster_uri = self.params.ocean_raster_uri

        ocean_vector = get_ocean_vector(ocean_vector_uri)
        water_raster = get_water_raster(ocean_raster_uri, shore_buffer)

        tiles = self.get_tiles()

        print(
            f"Processing region {self.params.region_id} "
            f"({len(tiles)} tiles of "
            f"{self.params.tile_dx}x{self.params.tile_dy} deg)"
        )

        self.scenes = list()

        for k, tile in enumerate(tiles):

            tile_id = self.get_tile_id(k)
            self.params.scene_id = tile_id
            self.scenes.append(tile_id)

            # Return: ee.List(max_num_images)
            image_list = get_images(
                date1,
                date2,
                ocean_vector,
                tile,
                satellite,
                orbit,
                max_num_images,
            )

            num_imgs = image_list.size().getInfo()

            if num_imgs < min_num_images:
                continue

            print(f"{num_imgs} scenes in tile {tile_id}")

            # Single band VH in BS (raster)
            composite = get_composite(image_list, min_num_images)
            composite = mask_water(composite, water_raster)
            composite = clip_polygon(composite, tile)

            self.detect(composite).export()

        # One param file per region (not per tile)
        self.check(every=check_every).save(folder)

        return self


# TODO: Shrink this class!
class DetectorInfraShore(DetectorInfra):
    """Run CFAR detector on a composite with custom geometry."""

    def process(self, check_every=60, folder=None, skip_done=False):
        """Run detect().export() on a composite for a region.

        Input: a date range, a region, and specific params.
        Output: a list of GEE tasks returned by export() to GCS.

        """
        self.params.replace(run_dir=folder)

        if skip_done and self.param_exists():
            print("Skipping", self.params.filename)
            return self

        date1, date2 = self.params.window_date
        satellite = self.params.satellite
        orbit = self.params.orbit
        # shore_buffer = self.params.shore_buffer
        min_num_images = self.params.min_num_images  # per pixel
        max_num_images = self.params.max_num_images  # in collection
        ocean_vector_uri = self.params.ocean_vector_uri
        # ocean_raster_uri = self.params.ocean_raster_uri

        ocean_vector = get_ocean_vector(ocean_vector_uri)
        # water_raster = get_water_raster(ocean_raster_uri, shore_buffer)

        tiles = self.get_tiles()

        print(
            f"Processing region {self.params.region_id} "
            f"({len(tiles)} tiles "
            f"{self.params.tile_dx}x{self.params.tile_dy} deg)"
        )

        self.scenes = []

        for k, tile in enumerate(tiles):

            tile_id = self.get_tile_id(k)
            self.params.scene_id = tile_id
            self.scenes.append(tile_id)

            # ee.List(max_num_images)
            image_list = get_images(
                date1,
                date2,
                ocean_vector,
                tile,
                satellite,
                orbit,
                max_num_images,
            )

            # TODO: Check for tiles without images and continue?
            # print(ee.length(image_list))
            # if len(image_list) < min_num_images:
            #     print('Not enought scenes in this region:', region_id)
            #     return self

            # Single band VH in BS (raster)
            composite = get_composite(image_list, min_num_images)
            # composite = mask_water(composite, water_raster)
            composite = clip_polygon(composite, ocean_vector)
            composite = clip_polygon(composite, tile)

            self.detect(composite).export()

        # One param file per region (not per tile)
        self.check(every=check_every).save(folder)

        return self
