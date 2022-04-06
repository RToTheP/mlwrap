from collections import Counter
import logging

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from mlwrap.dto import MLSettings
from mlwrap.enums import FeatureType, ProblemType


def resample_data(data, settings: MLSettings, problem_type: ProblemType):
    # SMOTE-NC is not designed to work only with categorical features. It requires some numerical features.
    cat_feature_count = len(
        [
            feature
            for feature in settings.features
            if feature.feature_type == FeatureType.Categorical and feature.active
        ]
    )
    num_features_exists = cat_feature_count != len(settings.features)

    if (
        problem_type == ProblemType.Classification
        and settings.balance_data_via_resampling
    ):
        if num_features_exists:
            data = resample_with_smotenc(data, settings)
        else:
            data = random_undersample(data, settings)
    return data


def resample_with_smotenc(data, settings: MLSettings):
    model_data = data[settings.model_feature_id]
    print("Original data samples per class ", Counter(model_data))

    cat_col_indexes = [
        data.columns.get_loc(feature.id)
        for feature in settings.features
        if feature.feature_type == FeatureType.Categorical and feature.active
    ]

    sm = SMOTENC(categorical_features=cat_col_indexes)
    data_resampled, model_data_resampled = sm.fit_resample(data, model_data)

    print("Resampled data samples per class ", Counter(model_data_resampled))

    return data_resampled


def random_undersample(data, settings: MLSettings):
    model_data = data[settings.model_feature_id]
    logging.debug(f"Original dataset shape {Counter(model_data)}")
    resampler = RandomUnderSampler()
    data_resampled, model_data_resampled = resampler.fit_resample(data, model_data)
    logging.debug(f"Resampled dataset shape {Counter(model_data_resampled)}")
    return data_resampled
