from typing import Union

from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from mlwrap.config import MLConfig
from mlwrap.enums import FeatureType, ProblemType


def get_resampler(df, config: MLConfig, problem_type: ProblemType):
    # SMOTE-NC is not designed to work only with categorical features. It requires some numerical features.
    numerical_features_exist = False
    categorical_features_exist = False
    categorical_column_indices = None
    if config.features is not None and len(config.features) > 0:
        numerical_features_exist = any(
            feature
            for feature in config.features.values()
            if feature.feature_type == FeatureType.Continuous
            and feature.id != config.model_feature_id
        )
        categorical_column_indices = [
            df.columns.get_loc(feature.id)
            for feature in config.features.values()
            if feature.feature_type == FeatureType.Categorical
            and feature.id != config.model_feature_id
        ]
        categorical_features_exist = any(categorical_column_indices)
    else:
        numerical_features_exist = any(df.dtypes.index != "category")
        categorical_column_indices = df.dtypes == "category"
        categorical_features_exist = any(categorical_column_indices)

    resampler = None
    if (
        problem_type == ProblemType.Classification
        and config.balance_data_via_resampling
    ):
        if numerical_features_exist:
            if categorical_features_exist:
                resampler = SMOTENC(categorical_features=categorical_column_indices)
            else:
                resampler = SMOTE()
        else:
            resampler = RandomUnderSampler()

    return resampler


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
    rng = np.random.default_rng()
    return rng.choice(data, size=n_samples, replace=False)
