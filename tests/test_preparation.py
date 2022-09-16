import pandas as pd
import pytest

from mlwrap import preparation
from mlwrap.config import MLConfig
from tests.datasets import IrisDataset


def test_split_data(iris: IrisDataset):
    # arrange
    df = pd.concat([iris.X, iris.y], axis=1)

    config = MLConfig(model_feature_id=iris.model_feature_id)

    # act
    X_train, X_test, y_train, y_test = preparation.split_data(
        df,
        model_feature_id=config.model_feature_id,
        train_size=config.train_test_split,
        shuffle=config.shuffle_before_splitting,
        problem_type=config.problem_type,
    )
    # assert
    assert pytest.approx(df.shape[0] * config.train_test_split) == X_train.shape[0]
    assert pytest.approx(df.shape[0] * (1 - config.train_test_split)) == X_test.shape[0]


def test_split_data_no_data(iris: IrisDataset):
    # arrange
    config = MLConfig(model_feature_id=iris.model_feature_id)

    # act and assert
    with pytest.raises(ValueError):
        preparation.split_data(
            None,
            model_feature_id=config.model_feature_id,
            train_size=config.train_test_split,
            shuffle=config.shuffle_before_splitting,
            problem_type=config.problem_type,
        )


def test_get_class_ratios_iris(iris: IrisDataset):
    # arrange
    class_ratios_ = preparation.get_class_ratios(iris.y)

    # assert
    one_third = 1.0 / 3.0
    assert all(cr == one_third for cr in class_ratios_.values())
