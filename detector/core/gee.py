"""GEE functions.

"""
import math
from datetime import datetime, timedelta
from pathlib import Path

import ee
import numpy as np
import pandas as pd

# Uncomment to set a different account
# ee.Authenticate()
# ee.Initialize()


def get_image(scene_id, collection="COPERNICUS/S1_GRD"):
    """Get GEE image object from scene ID."""
    path = ee.String((Path(collection) / scene_id).as_posix())
    return ee.Image(path)


def get_ocean_vector(ocean_vector_uri):
    """Ocean polygon to identify scenes that touch the ocean."""
    return ee.FeatureCollection(ocean_vector_uri)


def get_water_raster(ocean_raster_uri, shore_buffer):
    """Water raster to mask ocean area minus buffer.

    Invert land and ocean values, so ocean=1 and land=0
    Distance from shore raster (b1 = 1km, b2 = 2km, etc)
    """
    ocean_raster = ee.Image(ocean_raster_uri).subtract(1)
    return ocean_raster.where(ocean_raster.eq(-1), 1).select(
        f"b{shore_buffer}"
    )


def get_scenes_over_ocean(
    date,
    ocean_vector,
    satellite="",
    collection="COPERNICUS/S1_GRD",
):
    """Get one full day of imagery over ocean areas.

    Return list of scene IDs.

    """
    next_day = (
        datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    inputCollection = (
        ee.ImageCollection(collection)
        .filterBounds(ocean_vector)
        .filterDate(date, next_day)
        .filterMetadata("instrumentMode", "equals", "IW")
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            )
        )
        .filter(ee.Filter.eq("resolution", "H"))
    )
    # Get image ids that overlap with the ocean: gee -> pandas -> list
    gee_l = ee.List(inputCollection.aggregate_array("system:index"))
    df = pd.DataFrame({"scene_id": gee_l.getInfo()})
    scenes = list(df["scene_id"])
    if satellite:
        scenes = [s for s in scenes if s[:3] == satellite]
    return scenes


def calc_area(f):
    """Add area column to polygons.

    Used to sort with largest first, from which outer-ring is isolated.
    This is a workaround to get the outer ring of a multipolygon.
    """
    a = f.area(maxError=50)
    fa = f.set("area", a)
    return fa


# def clip_edge(polygon):
#     """Negative buffer for scene and composite extent.
#
#     Also smoothes polygon edges.
#     Used in BNR function.
#     Buffers in a polygon by 1km.
#     Also reduces the number of points and makes the polygon smaller.
#     """
#     return polygon.buffer(-1000)


def clip_edge(buffer=1000):
    """Negative buffer for scene and composite extent.

    Also smoothes polygon edges.
    Used in BNR function.
    Buffers in a polygon by `buffer` meters.
    Reduces the number of points and makes the polygon smaller.

    NOTE: Nested function to map func with argument.
    """

    def _clip_edge(polygon):
        return polygon.buffer(-buffer)

    return _clip_edge


def to_backscatter(image):
    """Convert dB from S1_GRD collection to backscatter units.

    Also multiply result by 1e5.
    Angle is between the pixel (line-of-sight) and the satellite's nadir.

    NOTE: Check with Christian Thomas at SkyTruth for the reference
    to this equation

    """
    angle_corrected = image.select(
        "V.|H."
    ).subtract(  # this should just be 'VH'
        image.select("angle")
        .multiply(math.pi / 180.0)
        .cos()
        .pow(2)
        .log10()
        .multiply(10.0)
    )
    backscatter = (
        ee.Image(10.0)
        .pow(angle_corrected.divide(10.0))
        .multiply(10000)
        # .select("VH")
    )
    return backscatter


def clip_polygon(img, region):
    """Region object -> EE Polygon -> clip."""
    if region is None:
        return img
    if isinstance(region, ee.featurecollection.FeatureCollection):
        region = region.geometry()
    if not isinstance(region, ee.geometry.Geometry):
        region = ee.Geometry.Polygon(region)
    return img.clip(region)


def to_geometry(region):
    """Convert Python coord list -> EE Polygon.

    A valid EE region has four coordinates, starting
    from the upper left and going counter-clockwise.

    """
    if not isinstance(region, ee.geometry.Geometry):
        region = np.squeeze(region).tolist()
        region = ee.Geometry.Polygon(region)
    return region


def get_image_footprint(
    image,
    water_raster,
    resolution=20,
    scale=500,
    percentile=1,
    buffer=0,
    usevv=False,
):
    """Removes border noise.

    NOTE: We've modified the original approach.

    This is border noise removal, it looks at VH band (only),
    and finds a low cutoff then it vectorizes it, and finds
    the outer ring of all the polygons.

    A visualization is here:

    https://bit.ly/36go0Jb

    The purple is what we are eliminating here.
    Across the scene, identify pixels in the Nth percentile.

    Notes
    -----
    This is now probably too agressive... Anything past March 2018
    needs to have this lowered to something like 1st percentile
    instead of the 5th percentile. To read more see this google doc:

    (Changed on Jul 22, 2021: Fernando P.)
    The strategy to compute the scene polygon now uses only the
    VH band, the 1st percentile, and reduced clip edge to 500m.

    https://bit.ly/3kV6xdj

    """
    percentiles = (
        image.select(["VH", "VV"])
        .mask(water_raster)
        .reduceRegion(
            reducer=ee.Reducer.percentile([percentile]),
            geometry=image.geometry(),
            scale=resolution,  # calc perc at full res
            maxPixels=1e11,
        )
    )
    # creating a number
    VH_perc = ee.Number(percentiles.get("VH"))
    VV_perc = ee.Number(percentiles.get("VV"))

    # reproject image to 500m to speedup, could probably be coarser
    downsampled = image.select(["VH", "VV"]).reproject(
        crs="EPSG:4326", scale=scale  # mask perc at low res
    )

    # Get everyting that is above the Nth perc
    downsampled_threshold = downsampled.select("VH").gt(VH_perc)

    if usevv:
        downsampled_threshold = downsampled_threshold.updateMask(
            downsampled.select("VV").gt(VV_perc)
        )

    mask_raster_2d = (
        downsampled_threshold.select("VH")
        .eq(1)
        .addBands(downsampled_threshold.select("VH").eq(1))
        .select(["VH", "VH_1"])
        .int()
    )
    full_footprint = mask_raster_2d.reduceToVectors(
        reducer=ee.Reducer.first(),
        geometry=image.geometry(),
        geometryType="polygon",
        maxPixels=1e11,
    ).filterMetadata("label", "equals", 1)

    # Exclude inner (smaller) polygons
    largest_polygon = (
        full_footprint.map(calc_area).sort("area", False).first()
    )
    coords = largest_polygon.geometry().coordinates().get(0)
    external_ring = ee.FeatureCollection(
        [ee.Geometry.Polygon([coords])]
    )
    if buffer:
        external_ring = external_ring.map(clip_edge(buffer))
    return external_ring


def get_ocean_footprint(image, ext_ring, water_raster, scale=500):
    """Make ocean_footprint.

    Mask the interesection between scene footprint and water.

    Notes:
        water_raster already contains a 1km buffer,
        clip_edge adds an extra buffer on top of this [removed].
    """
    # paint binary with ext_ring feature collection
    ext_ring_pixels = ee.Image().byte().paint(ext_ring, 1)
    ocean_pixels = ext_ring_pixels.updateMask(water_raster).addBands(
        ext_ring_pixels.updateMask(water_raster)
    )
    ocean_footprint = (
        (
            ocean_pixels.reduceToVectors(
                reducer=ee.Reducer.first(),
                geometry=image.geometry(),
                geometryType="polygon",
                scale=scale,
                maxPixels=1e11,
            )
        ).filterMetadata("label", "equals", 1)
        # .map(clip_edge)  # NOTE: removing from now
    )
    return ocean_footprint


def moving_mean(img, window=100, band="VH"):
    return img.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.square(
            radius=window, units="meters", normalize=False
        ),
        optimization="boxcar",
    )


def get_mean_donut(
    backscatter_mean_outer,
    backscatter_mean_inner,
    window_outer,
    window_inner,
):
    """Construct a donut-window mean from two square mean rasters."""
    outer_area = window_outer * window_outer
    inner_area = window_inner * window_inner
    donut_area = outer_area - inner_area
    return (
        backscatter_mean_outer.multiply(ee.Image(outer_area))
        .subtract(
            backscatter_mean_inner.multiply(ee.Image(inner_area))
        )
        .divide(ee.Image(donut_area))
    )


def get_connected_pixels(
    candidate_pixels, eight_connected=True, max_size=4, gte=2
):
    # Get mask with connected pixels
    connected_pixel_count = candidate_pixels.mask(
        candidate_pixels
    ).connectedPixelCount(
        maxSize=max_size, eightConnected=eight_connected
    )

    # Update candidates with connected pixels mask
    candidate_pixels = connected_pixel_count.gte(gte).updateMask(
        connected_pixel_count.gte(gte)
    )
    return candidate_pixels


# NOTE: Check if geometry can be obtained from `raster` instead of image
def get_centroids(raster, geometry, ocean_footprint, resolution):
    centroids = (
        raster.reduceToVectorsStreaming(
            reducer=ee.Reducer.first(),
            geometry=geometry,  # FIXME: check this is not
            scale=resolution,  # ocean_footprint.geometry()
            geometryType="centroid",
            eightConnected=True,
            bestEffort=False,
            # tileScale=4,  # works well with 16
            maxPixels=1e13,
        )
        .filterMetadata("label", "equals", 1)
        .filterBounds(ocean_footprint)
    )
    return centroids


# --- Specific to Fixed Infrastructure --- #


def get_images(
    date1,
    date2,
    ocean_vector,
    region,
    satellite="",
    orbit="",
    max_num_images=40,
    collection="COPERNICUS/S1_GRD",
):
    """Extract all images that touch a geometry between dates.

    Return list of images -> ee.List(max_num_images)

    """
    pol = "transmitterReceiverPolarisation"
    orb = "orbitProperties_pass"

    imgs = (
        ee.ImageCollection(collection)
        .filterDate(date1, date2)
        .filterBounds(ocean_vector)
        .filterBounds(region)
        .filterMetadata("instrumentMode", "equals", "IW")
        .filter(ee.Filter.eq("resolution", "H"))
        .filter(ee.Filter.listContains(pol, "VV"))
        .filter(ee.Filter.listContains(pol, "VH"))
    )

    if orbit.upper() == "A":
        imgs = imgs.filter(ee.Filter.eq(orb, "ASCENDING"))
    elif orbit.upper() == "D":
        imgs = imgs.filter(ee.Filter.eq(orb, "DESCENDING"))

    return (
        imgs.randomColumn("random_col", 1001)
        .sort("random_col")
        .limit(max_num_images)
        .toList(max_num_images)
    )


def count_images(image_list):
    """Return mosaic with image count per pixel."""
    return (
        ee.ImageCollection(image_list)
        .select("VH")
        .reduce(ee.Reducer.count())
    )


def get_composite(image_list, min_num_images=5):
    """Compute a composite from a list of images."""
    image_count = count_images(image_list)
    return (
        ee.ImageCollection(image_list)
        .map(to_backscatter)
        .select("VH")
        .median()
        .updateMask(image_count.gte(min_num_images).eq(1))
    )


def get_composite_raw(image_list):
    """Don't convert to backscatter neither remove border."""
    imgs = ee.ImageCollection(image_list).select(["VH", "VV"])
    return imgs.median()


def mask_water(img, water_raster):
    return img.updateMask(water_raster)
