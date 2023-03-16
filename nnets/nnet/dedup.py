from collections import defaultdict
from haversine import haversine

KM_DEG_LAT = 111


def _close_to_one_of(seq, item, tol_km):
    for other in seq:
        if haversine((item.lat, item.lon), (other.lat, other.lon)) <= tol_km:
            return True
    return False

def dedup(items, tol_km=0.04):
    """

    Parameters
    ----------
    items : iterable of item (namedtuple, itertuple, or similar)
        item.scene_id : str
        item.lon : float
        item.lat : float
    tol_km : float, optional
    """
    scale = KM_DEG_LAT / tol_km
    def key(x, lon_off, lat_off):
        timestr = x.scene_id.split('_')[2]
        # Rounding lat/lon off to 3 digits is ~1/10 of km -> 10 pixels at 10m resolution
        # Still could have edge cases where point fall on either side of grid, but not going
        # to worry about that for now.
        return (timestr, round(x.lon * scale) + lon_off, round(x.lat * scale) + lat_off)
    existing = defaultdict(list)
    dedupped = []
    for x in items:
        k = key(x, 0, 0)
        if k in existing and _close_to_one_of(existing[k], x, tol_km):
            continue
        for off1 in [-1, 0, 1]:
            for off2 in [-1, 0, 1]:
                k1 = key(x, off1, off2)
                existing[k].append(x)
        dedupped.append(x)
    return dedupped
