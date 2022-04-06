from functools import total_ordering
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from mlwrap.data.encoders import get_fitted_encoders, transform
from mlwrap.data.sampling import resample_data
from mlwrap.dto import MLSettings, DataDetails
from mlwrap.enums import DataType, ProblemType


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


def get_data(settings: MLSettings) -> pd.DataFrame:
    input_data = settings.input_data
    if input_data is None:
        logging.error("Input data missing")
        return None

    if input_data.data_type == DataType.DataFrame:
        return input_data.data_frame
    elif input_data.data_type == DataType.Csv:
        if input_data.data_path is None:
            logging.error("Csv data path missing")
            return None
        return pd.read_csv(input_data.data_path)
    raise NotImplementedError


def prepare_data(settings: MLSettings) -> DataDetails:
    df = get_data(settings)
    if df is None:
        return None

    encoders = get_fitted_encoders(df, settings)

    problem_type = settings.get_problem_type()
    train, test = split_data(
        df,
        model_feature_id=settings.model_feature_id,
        train_size=settings.train_test_split,
        shuffle=settings.shuffle_before_splitting,
        problem_type=problem_type,
    )

    # only resample the training set
    train = resample_data(train, settings, problem_type)

    total_row_count: int = train.shape[0] + test.shape[0]

    train_output = train.pop(settings.model_feature_id)
    train_input, encoded_feature_indices = transform(train, settings, encoders)
    train_output, _ = transform(train_output, settings, encoders)

    test_output = test.pop(settings.model_feature_id)
    test_input, _ = transform(test, settings, encoders)
    test_output, _ = transform(test_output, settings, encoders)

    return DataDetails(
        encoders=encoders,
        train_input=train_input,
        train_output=train_output,
        test_input=test_input,
        test_output=test_output,
        total_row_count=total_row_count,
        encoded_feature_indices=encoded_feature_indices,
    )
