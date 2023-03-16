# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# import pyperclip as pc
import time
from datetime import datetime, timedelta

import matplotlib.cm as cm
import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt

# %%
import numpy as np
import pandas as pd
import proplot as pplt
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
import seaborn as sn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_validate, train_test_split


# %%
# select training features and assign labels and features
def feat_labels(df, vessel = True):

    features = df[
        [
            'detect_id',
         'non_fishing', #'non_fishing_2', 'non_fishing_5', 'non_fishing_10',
            'non_fishing_under50m',
#          'cargo_tanker', 'cargo_tanker_2', 'cargo_tanker_5', 'cargo_tanker_10',
            'detections_km2_5km', 'detections_km2_5km_radius',
       'detections_km2_10km_radius', 'detections_km2_20km_radius',
            'length_m',
       'length_m_ave_5km', 'length_m_ave_5km_radius',
       'length_m_ave_10km_radius', 'length_m_ave_20km_radius',
       'length_m_stddev_5km', 'length_m_stddev_5km_radius',
       'length_m_stddev_10km_radius', 'length_m_stddev_20km_radius',
       'slope2_km', 'slope4_km', 'slope6_km', 'slope8_km', 'slope10_km',
       'slope12_km', 'slope14_km', 'slope16_km', 'slope18_km', 'slope20_km',
       'elevation_m', 'elevation_m_2km', 'elevation_m_4km',
       'elevation_m_stddev_2km', 'elevation_m_stddev_4km',
       'distance_from_shore_m', 'distance_from_port_m'
        ]
    ]

    # Saving feature names for later use
    feature_list = list(features)

    # assign labels and features for either vessel class
    # or effort class
    if vessel:
        labels = df.loc[:, "vessel_type"]
    else:
        labels = df.loc[:, "is_fishing"]

    return feature_list, labels, features


# run k-folds validation - default to 10
def k_folds(rf, train_features, train_labels, cv=10):
    cv = cross_validate(rf, train_features, train_labels, cv=5)
    accuracy = round(cv['test_score'].mean(),2)
    sd = round(cv['test_score'].std(),2)
    print(f"""Mean accuracy {accuracy}""")
    print(f"""Standard deviation {sd}""")
    return accuracy, sd


# fit and predict rf model
def fit_rf(rf, train_features, train_labels, test_features):
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    predictions_prob = rf.predict_proba(test_features)

    rf_out = pd.DataFrame(predictions_prob, columns=rf.classes_)
    rf_out["Prediction"] = predictions
    return rf_out


# report the accuracy metrics and generate confusion matrix plot
def report_accuracy(rf, test_labels, predictions, OOB = True):
    n_predictions = len(predictions)
    count = np.count_nonzero(predictions)

    n_fishing_count = predictions.value_counts()['non-fishing']
    fishing_count = predictions.value_counts()['fishing']

    print(f"Percentage of non-fishing vessels: {round((n_fishing_count)/n_predictions,2)}")
    print(f"Percentage of fishing vessels: {round((fishing_count)/n_predictions,2)}")

    # Print out of bag score
    if OOB:
        print(f"OOB score: {round(rf.oob_score_,2)}\n")

    print(f"{classification_report(test_labels, predictions)}\n")

    # View confusion matrix for test data and predictions
    fig, ax = plt.subplots(figsize=(6, 6))
    matrix = confusion_matrix(test_labels, predictions, normalize = 'pred')#, normalize = 'true')
    cm_display = ConfusionMatrixDisplay(matrix, display_labels=rf.classes_)
    cm_display.plot(ax=ax)
    plt.tight_layout()
    plt.show()

# report the accuracy metrics and generate confusion matrix plot
def report_accuracy_int(rf, test_labels, predictions):
    n_predictions = len(predictions)
    count = np.count_nonzero(predictions)

    n_fishing_count = predictions.value_counts()[0]
    fishing_count = predictions.value_counts()[1]

    print(f"Percentage of non-fishing vessels: {round((n_fishing_count)/n_predictions,2)}")
    print(f"Percentage of fishing vessels: {round((fishing_count)/n_predictions,2)}")

    # Print out of bag score
    print(f"OOB score: {round(rf.oob_score_,2)}\n")

    print(f"{classification_report(test_labels, predictions)}\n")

    # View confusion matrix for test data and predictions
    fig, ax = plt.subplots(figsize=(6, 6))
    matrix = confusion_matrix(test_labels, predictions)
    cm_display = ConfusionMatrixDisplay(matrix, display_labels=rf.classes_)
    cm_display.plot(ax=ax)
    plt.tight_layout()
    plt.show()


# plot mean decrease in impurity plot
def mdi_importance(rf, train_features, feature_list):
    importances = rf.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in rf.estimators_], axis=0
    )

    feature_names = [f"feature {i}" for i in range(train_features.shape[1])]

    forest_importances = pd.Series(importances, index=feature_list)

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.rcParams["font.size"] = 8
    forest_importances.plot.barh(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_xlabel("Mean decrease in impurity")
#     plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def perm_importance(rf_model, test_features, test_labels, feature_list):
    fig, ax = plt.subplots(figsize=(8, 5))
    result = permutation_importance(
        rf_model,
        test_features,
        test_labels,
        n_repeats=10,
        random_state=42,
        n_jobs=2,
    )
    forest_importances = pd.Series(result.importances_mean, index=feature_list)
    forest_importances.plot.barh(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_xlabel("Mean accuracy decrease")
    plt.rcParams["font.size"] = 8
    # plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()


# funtion to split test and train data manually
def manual_split(df_train, df_test):

    # Define train features
    train_features = df_train[
        [
           'cargo_tanker',
         'detections_km2_5km',
         'length_m',
         'length_m_ave_5km',
         'length_m_stddev_5km',
         'length_m_stddev_5km_radius',
         'length_m_stddev_10km_radius',
#          'length_m_stddev_20km_radius',
         'slope2_km',
         'elevation_m',
         'distance_from_shore_m',
         'distance_from_port_m'
        ]
    ]
    # saving feature names for later use
    feature_list = list(train_features)

    # assign labels and features
    train_labels = df_train.loc[:, "vessel_type"]

    # define test features
    test_features = df_test[
        [
            'cargo_tanker',
         'detections_km2_5km',
         'length_m',
         'length_m_ave_5km',
         'length_m_stddev_5km',
         'length_m_stddev_5km_radius',
         'length_m_stddev_10km_radius',
#          'length_m_stddev_20km_radius',
         'slope2_km',
         'elevation_m',
         'distance_from_shore_m',
         'distance_from_port_m'
        ]
    ]

    # assign labels and features
    test_labels = df_test.loc[:, "vessel_type"]

    return (train_features,
        test_features,
        train_labels,
        test_labels,
        feature_list)
