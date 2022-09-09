import logging
from typing import Dict, List, Tuple
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from mlwrap.config import CleaningReport, MLConfig
from mlwrap.data.cleaning import clean_inference_data, clean_training_data
from mlwrap.data.config import DataDetails
from mlwrap.data.encoders import get_fitted_encoders, transform
from mlwrap.data.sampling import get_background_data, resample_data
from mlwrap.io import load_pkl
from mlwrap.enums import CleaningType, ProblemType, Status


def split_data_x_y(
    data: pd.DataFrame,
    model_feature_id: str,
    train_size: float,
    shuffle: bool,
    problem_type: ProblemType,
):
    X_train, X_test = split_data(
        data=data,
        model_feature_id=model_feature_id,
        train_size=train_size,
        shuffle=shuffle,
        problem_type=problem_type,
    )
    y_train = X_train.pop(model_feature_id)
    y_test = X_test.pop(model_feature_id)
    return X_train, X_test, y_train, y_test


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

    split_data = train_test_split(
        data, train_size=train_size, shuffle=shuffle, stratify=labels
    )
    return split_data


def prepare_training_data(config: MLConfig, df: pd.DataFrame()) -> DataDetails:
    if df is None:
        return DataDetails(status=Status.invalid_data)

    tup: Tuple[Status, pd.DataFrame, CleaningReport] = clean_training_data(df, config)
    status: Status = tup[0]
    data: pd.DataFrame = tup[1]
    cleaning_report: CleaningReport = tup[2]
    if status != Status.success:
        return DataDetails(status=status)

    encoders = get_fitted_encoders(df, config)

    problem_type = config.problem_type
    train, test = split_data(
        df,
        model_feature_id=config.model_feature_id,
        train_size=config.train_test_split,
        shuffle=config.shuffle_before_splitting,
        problem_type=problem_type,
    )

    # only resample the training set
    train = resample_data(train, config, problem_type)

    if problem_type == ProblemType.Classification:
        class_weights = get_class_weights(train, config)
        class_ratios: Dict[str, float] = get_class_ratios(train, config)
    else:
        class_weights = class_ratios = None
    total_row_count: int = train.shape[0] + test.shape[0]

    train_output = train.pop(config.model_feature_id)
    train_input, encoded_feature_indices = transform(train, config, encoders)
    train_output, _ = transform(train_output, config, encoders)

    test_output = test.pop(config.model_feature_id)
    test_input, _ = transform(test, config, encoders)
    test_output, _ = transform(test_output, config, encoders)

    background_data = get_background_data(train_input, config)

    return DataDetails(
        status=status,
        cleaning_report=cleaning_report,
        encoders=encoders,
        train_input=train_input,
        train_output=train_output,
        test_input=test_input,
        test_output=test_output,
        class_weights=class_weights,
        class_ratios=class_ratios,
        total_row_count=total_row_count,
        encoded_feature_indices=encoded_feature_indices,
        background_data=background_data,
    )


def prepare_inference_data(config: MLConfig, data: pd.DataFrame()) -> DataDetails:
    if data is None:
        return DataDetails(status=Status.invalid_data)

    # remove the model feature if it is in the inference data
    if config.model_feature_id in data.columns:
        data.pop(config.model_feature_id)

    tup: Tuple[Status, pd.DataFrame, CleaningReport] = clean_inference_data(
        data, config
    )
    status: Status = tup[0]
    data: pd.DataFrame = tup[1]
    cleaning_report: CleaningReport = tup[2]
    if status != Status.success:
        return DataDetails(
            status=Status.invalid_data,
            cleaning_report=cleaning_report,
        )

    unknown_values_input_rows: List[str] = [
        x.row
        for x in cleaning_report.cleaning_records
        if x.cleaning_type == CleaningType.row_feature_out_of_range
    ]

    # load encoders
    encoders = (
        load_pkl(config.encoder_bytes) if config.encoder_bytes is not None else None
    )

    # get a list of remaining inference_id's
    inference_ids: List[str] = data.index

    # get transformed data
    inference_input, encoded_feature_indices = transform(data, config, encoders)

    # load the background data
    background_data = (
        load_pkl(config.background_data_bytes)
        if config.background_data_bytes is not None
        else None
    )

    return DataDetails(
        status=status,
        cleaning_report=cleaning_report,
        unknown_values_input_rows=unknown_values_input_rows,
        inference_input=inference_input,
        inference_ids=inference_ids,
        encoders=encoders,
        encoded_feature_indices=encoded_feature_indices,
        background_data=background_data,
    )


def get_class_ratios(data, config: MLConfig) -> Dict[str, float]:
    value_counts_ = pd.value_counts(data[config.model_feature_id])
    value_counts_.sort_values(ascending=False, inplace=True)
    total_counts = np.size(data[config.model_feature_id])
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


def get_validation_data(input, output):
    if (
        input is not None
        and output is not None
        and input.shape[0] > 0
        and output.shape[0] > 0
    ):
        return input, output
    return None


def flatten_output_data(problem_type: ProblemType, train_output, test_output=None):
    if problem_type == ProblemType.Classification:
        # classification problem so flatten the output array
        train_output = np.argmax(train_output, axis=1)
        if test_output is not None:
            test_output = np.argmax(test_output, axis=1)
    elif problem_type == ProblemType.Regression:
        train_output = np.squeeze(train_output, axis=1)
        if test_output is not None:
            test_output = np.squeeze(test_output, axis=1)

    return train_output, test_output
