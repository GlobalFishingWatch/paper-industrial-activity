"""Calls `DetectorVessel` to generate and export footprints only.

"""
from .detector import DetectorVessel
from .gee import get_image


class ExportFootprint(DetectorVessel):
    """Compute detection footprint and export to GCS.

    Same steps as DetectorVessel but skipping the detection part.

    """

    def process(self, folder=None, skip_done=True, check_every=60):
        self.params.replace(run_dir=folder)

        if skip_done and self.param_exists():
            print(f"Skipping {self.params.filename}, processed")
            return self

        date = self.params.date
        scenes = self.get_scenes()  # 1 scene or 1 day of scenes
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
            self.export(footprint)

        # One param file per day
        self.check(every=check_every).save(folder)

        return self
