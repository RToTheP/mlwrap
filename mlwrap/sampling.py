from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd

from mlwrap.config import MLConfig
from mlwrap.enums import FeatureType, ProblemType


def get_resampler(df, config: MLConfig, problem_type: ProblemType):
    # SMOTE-NC is not designed to work only with categorical features. It requires some numerical features.
    numerical_features_exist = False
    categorical_features_exist = False
    cat_feature_count = None
    categorical_column_indices = None
    if config.features is not None and len(config.features) > 0:
        numerical_features_exist = any(
            feature
            for feature in config.features.values()
            if feature.feature_type == FeatureType.Continuous and feature.active and feature.id != config.model_feature_id
        )
        categorical_column_indices = [
            df.columns.get_loc(feature.id)
            for feature in config.features.values()
            if feature.feature_type == FeatureType.Categorical and feature.active and feature.id != config.model_feature_id
        ]
        categorical_features_exist = any(categorical_column_indices)
    else:
        numerical_features_exist = any(df.dtypes.index != 'category')
        categorical_column_indices = df.dtypes == 'category'
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
