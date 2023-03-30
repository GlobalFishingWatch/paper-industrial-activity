"""Class to handle input parameters, files and table names.

"""
import re
import yaml
import json
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path

from .checks import check_name, check_prefix, check_suffix

CONFIG = "config.yaml"


class Params(object):
    """Parameters used by the Detector class and functions.

    Read all params from config.yaml or params.json file.
    Generate names (subbucket, dataset) dynamically from params.
    Replace/Add parameters from function arguments.
    Update all params and names automatically when needed.
    Provide methods to handle parameters and log files.

    """
    def __init__(self, config=CONFIG):

        # Read from YAML or JSON and set Attrs
        # self._set_attrs(self._read_file(config))
        self.__dict__.update(self._read_file(config))
        # NOTE: Generate subbucket only at save time
        # self.subbucket = self._get_subbucket()
        self.dataset = self._get_dataset()
        self.commit = self._get_commit()
        self._check_names()

    def _get_dataset(self):
        return f"{self.dataset_prefix}{self.version}"

    def _get_subbucket(self):
        comp_suffix = self.comp_suffix if self.comp_suffix else ""
        sat_suffix = f"_{self.satellite}" if self.satellite else ""
        orbit_suffix = f"_{self.orbit}" if self.orbit else ""
        return (
            f"{self.resolution}m_{self.thresholdx}x_"
            f"{self.window_inner}i_{self.window_outer}o_"
            f"{self.dialation_radius}d{comp_suffix.lower()}"
            f"{sat_suffix.lower()}{orbit_suffix.lower()}{self.suffix}"
        )

    def _get_commit(self):
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("ascii")
        )

    @property
    def subbucket_path(self):
        return f"{self.bucket}/{self.version}/{self.subbucket}"

    @property
    def seive_suffix(self):
        return "_s" if self.use_seive else "_ns"

    @property
    def stats_suffix(self):
        return "_stats" if self.include_stats else ""

    @property
    def filename(self):
        """Name of parameter file (JSON)."""
        prefix = self.param_prefix
        core = self.subbucket + "_" + self.date.replace("-", "")
        suffix = f"_{self.region_id}" if self.region_id else ""
        return f"{prefix}{core}{suffix}.json"

    def _read_file(self, pfile):
        msg = f"file extention must be .yaml or .json, got {pfile}"
        with open(pfile) as f:
            if pfile.endswith('.yaml'):
                return yaml.load(f, Loader=yaml.FullLoader)
            elif pfile.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError(msg)

    # NOTE: In case we want type/form check at
    # instantiation, we can add checks here (UNUSED).
    def _set_attrs(self, params):
        for key, val in params.items():
            setattr(self, key, val)

    def _get_date_from_scene_id(self):
        d = re.search('[0-9]{8}T[0-9]{6}', self.scene_id).group()
        return str(datetime.strptime(d, "%Y%m%dT%H%M%S").date())

    def _file_exists(self):
        """Check if current parameter file exists."""
        return (Path(self.run_dir) / self.filename).exists()

    def _check_names(self):
        check_prefix(self.dataset_prefix)
        check_prefix(self.table_prefix)
        check_prefix(self.param_prefix)
        check_suffix(self.suffix)
        check_suffix(self.comp_suffix)
        check_suffix(self.stats_suffix)
        check_suffix(self.seive_suffix)
        check_name(self.subbucket)
        check_name(self.dataset)

    def _update(self):
        """Regenerate dynamic params."""
        # self.commit = self._get_commit()  # NOTE: should we update?
        if not self.subbucket:
            # if not given, generate one
            self.subbucket = self._get_subbucket()
        if self.scene_id:
            # if processing scene instead of date
            self.date = self._get_date_from_scene_id()
        self._check_names()
        return self

    def add(self, **kwargs):
        """Add new params with values from kwargs."""
        for key, val in kwargs.items():
            if val is not None:
                setattr(self, key, val)
        self._update()
        return self

    def replace(self, **kwargs):
        """Replace existing params with values from kwargs."""
        for key, val in kwargs.items():
            assert hasattr(self, key), f"param {key} doesn't exist"
            if val is not None:
                setattr(self, key, val)
        self._update()
        return self

    def save(self, folder=None):
        """Save params dictionary to a JSON file."""
        self.replace(run_dir=folder)
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        fout = Path(self.run_dir) / self.filename
        with open(fout, "w") as f:
            json.dump(self.__dict__, f, indent=4)
            print("PARAMS ->", fout)
        return self

    def print(self):
        self._update()
        for attr, value in self.__dict__.items():
            print(attr, ":", value)
        return self

    @staticmethod
    def combine(*args):
        """Combine all args (iterables and/or single values)."""
        # Make all args iterable
        args = [
            [arg] if not isinstance(arg, (list, tuple)) else arg
            for arg in args
        ]
        for args in product(*args):
            yield args

    @staticmethod
    def combine_with_dict(config):
        """Combine all args (iterables and/or single values)."""
        for key, value in dict(config).items():
            if value is None:
                del config[key]
        for values in product(*config.values()):
            if values is not None:
                yield (dict(zip(config.keys(), values)))
