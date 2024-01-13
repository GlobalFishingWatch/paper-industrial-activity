import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# +
COLOR = '#0868ac'
FONT = 11
    
    
def get_lonlat(detect_id):
    scene_id, lon, lat = detect_id.split(';')
    return float(lon), float(lat)


def get_lonlats(detect_ids):
    lonlats = np.array([get_lonlat(uid) for uid in detect_ids])
    return lonlats[:,0], lonlats[:,1]


def subplot(f_train, f_test, s=0.1, n=331, title='', alpha=1):
    df_train = pd.read_csv(f_train)
    df_test = pd.read_csv(f_test)

    x_train, y_train = get_lonlats(df_train.detect_id)
    x_test, y_test = get_lonlats(df_test.detect_id)
    
    # Trick to have all maps with full extent
    x_train[0] = -180
    x_train[-1] = 180
    y_train[0] = 70
    y_train[-1] = 70

    plt.subplot(n)
    plt.scatter(x_train, y_train, s=s, c=COLOR, alpha=alpha)
    plt.scatter(x_test, y_test, s=s, c=COLOR, alpha=alpha)
    plt.legend(
        [f'Training ({len(x_train)})', f'Holdout ({len(x_test)})'],
        loc='lower left',
        frameon=False,
        title=title,
        markerscale=0,
        handletextpad=0,
        fontsize=FONT,
        title_fontsize=FONT,
    )
    ax = plt.gca()
    ax.axis('off')
    ax.set_facecolor("yellow")

# +
plt.figure(figsize=(11, 5 * 3))

f_train = '../data/vessels_train.csv.zip'
f_test = '../data/vessels_test.csv.zip'
subplot(f_train, f_test, s=0.5, n=311, title='Presence and length data')

f_train = '../data/fishing_train.csv.zip'
f_test = '../data/fishing_test.csv.zip'
subplot(f_train, f_test, s=0.005, n=312, title='Fishing and nonfishing data')

f_train = '../data/infra_train.csv.zip'
f_test = '../data/infra_test.csv.zip'
subplot(f_train, f_test, s=1.5, n=313, title='Offshore infrastructure data', alpha=0.75)

plt.subplots_adjust(wspace=0, hspace=0)

if 1:
    plt.savefig(
        './training_data_maps_v3.jpg',
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
# -


