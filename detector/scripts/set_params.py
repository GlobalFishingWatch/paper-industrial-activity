# -*- coding: utf-8 -*-
"""
Add new parameters or replace existing values in PARAMS files.

"""
import sys
import argparse

from params import Params


def get_args():
    parser = argparse.ArgumentParser(
        description="Add/Replace parameters in PARAMS files"
    )
    parser.add_argument(
        "files",
        nargs='+',
        help="PARAMS_*.json",
    )
    parser.add_argument(
        "-a",
        dest="add",
        help="Add params: key1=val1,key2=val2,..",
        default='',
    )
    parser.add_argument(
        "-r",
        dest="replace",
        help="Replace params: key1=val1,key2=val2,..",
        default='',
    )
    return parser.parse_args()


# params to add/replace
kwargs = {
    "invalid": False,
    "uploaded": False,
    "matched": False,
    "evaluated": False,
}

replace = True


def main(pfile, replace, kwargs):
    if replace:
        Params(pfile).replace(**kwargs).save()
    else:
        Params(pfile).add(**kwargs).save()


if __name__ == "__main__":

    args = get_args()
    files = args.files

    if args.replace:
        ss = args.replace
        replace = True
    else:
        ss = args.add
        replace = False

    pairs = [s.split("=") for s in ss.split(",")]
    kw = {k: eval(v) for (k, v) in pairs}
    [main(f, replace, kw) for f in files]
    print(f"Replaced/Added params: {kw}")
