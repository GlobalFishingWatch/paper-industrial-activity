import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
from recombinator.optimal_block_length import optimal_block_length
from recombinator.block_bootstrap import (
    stationary_bootstrap,
    circular_block_bootstrap,
)


def bootstrap_resample(values, column="fishing", samples=100000):
    values = values[values.rolling_date >= datetime(2018, 1, 1)]
    pre_2020_values = values[values.rolling_date < datetime(2020, 1, 1)][column].values
    post_2020_values = values[values.rolling_date >= datetime(2020, 1, 1)][column].values

    pre_2020_trend = 0.5 * (pre_2020_values[:365] + pre_2020_values[365:])
    
    delta_pre = np.concatenate(
        [
            pre_2020_values[:365] - pre_2020_trend,
            pre_2020_values[365:] - pre_2020_trend,
        ],
        axis=0,
    )
    delta_post = np.concatenate(
        [
            post_2020_values[:365] - pre_2020_trend,
            # Note skip leap day
            post_2020_values[366:] - pre_2020_trend[: len(post_2020_values[366:])],
        ],
        axis=0,
    )

    b_star_pre = optimal_block_length(delta_pre)
    b_star_post = optimal_block_length(delta_post)
    
    block_length = int(
        round(0.5 * (b_star_pre[0].b_star_sb + b_star_post[0].b_star_sb))
    )

    pre_bs = stationary_bootstrap(
        delta_pre, block_length=block_length, replications=samples
    )

    post_bs = stationary_bootstrap(
        delta_post, block_length=block_length, replications=samples
    )

    # print(delta_pre.shape, delta_post.shape)
    # print(delta_pre.max(), delta_post.max())
    # print(b_star_pre[0].b_star_sb, b_star_post[0].b_star_sb)
    
    plt.figure()
    plt.plot(pre_2020_values[:365])
    plt.plot(pre_2020_values[365:])
    plt.plot(pre_2020_trend)
    plt.title(f"Seasonal cycle {column}")
    plt.show()

    return pre_2020_trend, pre_bs[:, : post_bs.shape[1]], post_bs


# +
# The time series data files are created by the script
# FishingNonfishingSeries.py that generates Figure 3
files = ['ts_world.csv', 'ts_china.csv', 'ts_global.csv']
names = ['World', 'China', 'Global']

for n, f in zip(names, files):
    # Fishing
    data = pd.read_csv(f, parse_dates=["rolling_date"])
    trend, pre, post = bootstrap_resample(data, column='fishing')
    delta = post - pre
    plt.figure()
    plt.hist(100 * delta.mean(axis=1) / trend.mean(), bins=50)
    mean = 100 * delta.mean(axis=1).mean() / trend.mean()
    std = 100 * delta.mean(axis=1).std() / trend.mean()
    plt.title(f"{n} fishing (bootstrap) Mean={mean:.0f}%, SE={std:.0f}%")
    
    # Non-fishing
    data = pd.read_csv(f, parse_dates=["rolling_date"])
    trend, pre, post = bootstrap_resample(data, column='nonfishing')
    delta = post - pre
    plt.figure()
    plt.hist(100 * delta.mean(axis=1) / trend.mean(), bins=50)
    mean = 100 * delta.mean(axis=1).mean() / trend.mean()
    std = 100 * delta.mean(axis=1).std() / trend.mean()
    plt.title(f"{n} non-fishing (bootstrap) Mean={mean:.0f}%, SE={std:.0f}%")
# -


