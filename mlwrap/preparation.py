from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from mlwrap.enums import ProblemType
from mlwrap.config import MLConfig

def split_data(
    data: pd.DataFrame,
    model_feature_id: str,
    train_size: float,
    shuffle: bool,
    problem_type: ProblemType,
):
    if data is None:
        raise ValueError("Data missing")
    if train_size >= 1.0:
        return data

    labels = None
    if problem_type == ProblemType.Classification and shuffle:
        labels = data[model_feature_id]

    X_train, X_test = train_test_split(
        data, train_size=train_size, shuffle=shuffle, stratify=labels
    )
    y_train = X_train.pop(model_feature_id)
    y_test = X_test.pop(model_feature_id)
    return X_train, X_test, y_train, y_test

def get_class_ratios(y) -> Dict[str, float]:
    value_counts_ = pd.value_counts(y)
    value_counts_.sort_values(ascending=False, inplace=True)
    total_counts = np.size(y)
    class_ratio_ = {
        x: value_counts_[x] / int(total_counts) for x in value_counts_.index
    }
    return class_ratio_


def get_class_weights(data: pd.DataFrame, config: MLConfig) -> np.ndarray:
    # class weights can be used to adjust the loss to compensate for class imbalance in the data, e.g. an imbalance of 80:20 would mean we want weights of 1:4 (or 0.25:1 to keep weights lower)
    # e.g. classWeights = { 0:0.25, 1: 1.}
    column_data = data[config.model_feature_id]
    return class_weight.compute_class_weight(
        "balanced", classes=np.unique(column_data), y=column_data
    )
