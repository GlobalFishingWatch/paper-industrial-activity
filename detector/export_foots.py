"""Script to generate and export footprints, passing command-line args.

Notes
-----

- Edit 'subbucket' below.
- Use 'skip_done=False' if passing scene IDs instead of dates.
  If two scene IDs have the same date, only one params file is create.
  The upload script will retrieve both footprints from GCS (same dir).

Examples
--------

# passing individual scenes
scene_ids = [
    "S1A_IW_GRDH_1SDV_20180104T062704_20180104T062729_020001_02211F_22E3",
    "S1B_IW_GRDH_1SDV_20200201T051926_20200201T051951_020071_025FC9_324B",
    "S1A_IW_GRDH_1SDV_20200201T003833_20200201T003902_031052_039134_3DA3",
]

for scene_id in scene_ids:
    ...

"""
import sys
import textwrap

from core.footprint import ExportFootprint
from core.ranges import date_range

# --- EDIT ----------------------------------

# gs://gfw-sentinel-1-detections/{version}/{subbucket}/{YYYYMMDD}
subbucket = "footprints"

# save PARAMS to footprints_{YYYYMMDD}_{YYYYMMDD}
folder_prefix = "footprints_"

# -------------------------------------------

if len(sys.argv) < 3:
    msg = """Usage:

    export_foots.py date1 date2 sat
    export_foots.py date1 ndays sat

    where:
    date = YYYY-MM-DD
    sat = S1A | S1B | ignore (for both)

    output:
    footprints_date1_date2/
    """
    print(textwrap.dedent(msg))
    sys.exit()

date1 = sys.argv[1]
date2 = sys.argv[2]

d1, d2 = date1.replace('-', ''), date2.replace('-', '')

if len(sys.argv) > 3:
    sat = sys.argv[3]
else:
    # Will export both S1A and S1B to the same GCS bucket
    sat = ""

folder = f"{folder_prefix}{d1}_{d2}"

for date in date_range(date1, date2):
    ExportFootprint(
        date=date,
        satellite=sat,
        subbucket=subbucket,
        foot_scale=500,
        foot_buffer=500,
        foot_percentile=1,
        foot_usevv=False,
    ).process(folder=folder, skip_done=True)
    # ).save(folder=folder, skip_done=True)
