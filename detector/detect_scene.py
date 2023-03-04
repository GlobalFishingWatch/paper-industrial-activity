"""A template for single-scene detection.

This is a draft of how the steps should be implemented,
the functionalities below (A, B, C) need to be added.

"""
from core.detector import DetectorVessel
from core.ranges import date_range

# Folder to save PARAMS files
folder = "parameters"

# (A) Retrieve all scenes from GEE for a day
scene_list = check_available_scenes('YYYYMMDD')

# (B) Check if there are any unprocessed scenes
new_scenes = get_new_scenes(scene_list)

if new_scenes:

    for scene_id in new_scenes:
        DetectorVessel(
            scene_id=scene_id,
            satellite="S1A",
            suffix="_a",
            thresholdx=22,
            resolution=20,
            window_inner=200,
            window_outer=600,
            dialation_radius=60,
        ).process(folder=folder, skip_done=True)

        # (C) Report if processing is successful
        update_processed_scenes(scene_id)
