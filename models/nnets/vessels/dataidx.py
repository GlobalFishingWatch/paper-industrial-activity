"""
Generate train/val/test indices mapping to data in Zarr.

It samples data by days, so as to have unique days in the
validation and test sets (not included on any of the other sets).

"""
import sys
import numpy as np
import zarr

from .data import get_dates_from_ids, group_kfold, group_split

np.random.seed(123)

infile = sys.argv[1]

f = zarr.open(infile, "r")

detect_ids = f["detect_id"][:]

dates = get_dates_from_ids(detect_ids)
indices = np.arange(len(dates))

train_index, test_index = group_split(indices, dates, test_size=0.2)
index_pairs = group_kfold(train_index, dates[train_index], n_splits=5)

# For training with full data
val_index = np.random.choice(train_index, len(train_index) // 5)

idxfile = infile.replace('.zarr', '_index.zarr')
root = zarr.open(idxfile, 'w')
root.array('test', test_index)
root.array('train/all/train', train_index)
root.array('train/all/validation', val_index)

for k, (train_idx, val_idx) in enumerate(index_pairs):
    root.array(f"train/{k}/train", train_idx)
    root.array(f"train/{k}/validation", val_idx)

print(root.tree())
