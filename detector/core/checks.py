"""Various checks for (GFW) naming conventions.

"""
import json


def check_name(x):
    if "-" in x:
        raise ValueError(f"name must not contain '-', got {x}")


def check_prefix(x):
    check_name(x)
    if x and not x.endswith("_"):
        raise ValueError(f"prefix must end with '_', got {x}")


def check_suffix(x):
    check_name(x)
    if x and not x.startswith("_"):
        raise ValueError(f"suffix must start with '_', got {x}")


def is_valid_json(jfile):
    f = open(jfile)
    try:
        json.load(f)
    except ValueError as err:
        err
        return False
    return True


def validate_jsons(jfiles):
    jlist = [f for f in jfiles if is_valid_json(f)]
    print(f"{len(jlist)} valid GeoJSONs")
    return jlist
