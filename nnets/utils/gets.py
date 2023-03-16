import re
import datetime
import logging
import subprocess
import importlib
import numpy as np
from pathlib import Path

root_dir = Path(__file__).parents[1]


def get_log_dir(identifier, kind="runs", root='untracked'):
    if isinstance(identifier, (datetime.datetime, datetime.date)):
        identifier = f"{identifier:%Y%m%dT%H%M%S}"
    return Path(root) / kind / identifier


def get_run_dir(guild_run=True, uid=None, root='untracked'):
    if guild_run:
        run_dir = Path.cwd()
    elif isinstance(uid, (datetime.datetime, datetime.date)):
        uid = f"{uid:%Y%m%dT%H%M%S}"
        run_dir = Path(root) / "runs" / uid
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def current_commit(require_clean, not_clean_suffix="-not-clean"):
    not_clean = subprocess.check_output(["git", "status", "-s"]).strip()
    if not_clean:
        if require_clean:
            raise RuntimeError(
                "git repo must be clean (`--no-require-clean` to override)"
            )
        else:
            logging.warning(
                "git repo is not clean, but"
                " `--no-require-clean` specified so continuing"
            )
    commit = (
        subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"])
        .strip()
        .decode("ascii")
    )
    if not_clean:
        return f"{commit}{not_clean_suffix}"
    return commit


def setup_logging(log_dir, args, logfile):
    """Setup logging, including dumping logs to run dir"""
    _ = datetime.datetime.now().replace(microsecond=0).isoformat()
    level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError("Invalid log level: %s" % args.log_level)
    path = (log_dir / f"{logfile}").as_posix()
    logger = logging.getLogger()
    logger.setLevel(level)
    for x in logger.handlers:
        x.setLevel(logging.ERROR)
    fh = logging.FileHandler(path)
    fh.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.captureWarnings(True)


def find_most_recent_run(runs_dir):
    """Return path to most recent run directory."""
    candidates = [x for x in Path(runs_dir).glob("*") if x.is_dir()]
    return Path(sorted(candidates)[-1])


def get_model_module(config):
    root = Path(__file__).parent.stem
    module = config["model-name"]
    return importlib.import_module(f"{root}.{module}")


def import_module(module):
    root = Path(__file__).parent.stem
    return importlib.import_module(f"{root}.{module}")


def get_date_from_id(idstr):
    """Return first occurence of date pattern."""
    return re.search("[0-9]{8}T[0-9]{6}", idstr).group()[:8]


def get_date_from_ids(idstrs):
    return np.array([get_date_from_id(s) for s in idstrs])
