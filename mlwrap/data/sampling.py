from collections import Counter
import logging
from typing import Union

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from mlwrap.config import MLConfig
from mlwrap.enums import FeatureType, ProblemType


def resample_data(data, config: MLConfig, problem_type: ProblemType):
    # SMOTE-NC is not designed to work only with categorical features. It requires some numerical features.
    cat_feature_count = len(
        [
            feature
            for feature in config.features.values()
            if feature.feature_type == FeatureType.Categorical and feature.active
        ]
    )
    num_features_exists = cat_feature_count != len(config.features)

    if (
        problem_type == ProblemType.Classification
        and config.balance_data_via_resampling
    ):
        if num_features_exists:
            data = resample_with_smotenc(data, config)
        else:
            data = random_undersample(data, config)
    return data


def resample_with_smotenc(data, config: MLConfig):
    model_data = data[config.model_feature_id]
    print("Original data samples per class ", Counter(model_data))

    cat_col_indexes = [
        data.columns.get_loc(feature.id)
        for feature in config.features.values()
        if feature.feature_type == FeatureType.Categorical and feature.active
    ]

    sm = SMOTENC(categorical_features=cat_col_indexes)
    data_resampled, model_data_resampled = sm.fit_resample(data, model_data)

    print("Resampled data samples per class ", Counter(model_data_resampled))

    return data_resampled


def random_undersample(data, config: MLConfig):
    model_data = data[config.model_feature_id]
    logging.debug(f"Original dataset shape {Counter(model_data)}")
    resampler = RandomUnderSampler()
    data_resampled, model_data_resampled = resampler.fit_resample(data, model_data)
    logging.debug(f"Resampled dataset shape {Counter(model_data_resampled)}")
    return data_resampled


def get_background_data(
    data: Union[pd.DataFrame, np.ndarray], config: MLConfig, sample_type: str = "random"
) -> np.ndarray:
    if data.shape[0] <= config.explanation_background_samples:
        return data

    if sample_type == "random":
        return random_sample(data, config.explanation_background_samples)
    elif sample_type == "kmeans":
        return kmeans_sample(data, config.explanation_background_samples)
    else:
        raise NotImplementedError("Unknown sampling type: %s", sample_type)


def kmeans_sample(data: Union[pd.DataFrame, np.ndarray], n_samples: int) -> np.ndarray:
    sample = KMeans(n_clusters=n_samples, random_state=0).fit(data)

    for i in range(n_samples):
        for j in range(data.shape[1]):
            xj = data[:, j]
            ind = np.argmin(np.abs(xj - sample.cluster_centers_[i, j]))
            sample.cluster_centers_[i, j] = data[ind, j]

    return sample.cluster_centers_


def random_sample(data: Union[pd.DataFrame, np.ndarray], n_samples: int) -> np.ndarray:
    return data[np.random.choice(data.shape[0], n_samples, replace=False)]
