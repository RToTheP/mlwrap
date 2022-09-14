import unittest

import numpy as np
import pandas as pd

from mlwrap.config import MLConfig
from mlwrap import encoders
from mlwrap.enums import ProblemType
from tests.datasets import DiabetesDataset, IrisDataset


class TestCyclicalEncoder(unittest.TestCase):
    def test_inverse_transform(self):
        # arrange
        cyclical_period = 7
        data = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        encoder = encoders.CyclicalEncoder(cyclical_period)

        # act
        transformed_data = encoder.transform(data)
        inverse_transformed_data = encoder.inverse_transform(transformed_data)

        # assert
        for n in range(data.shape[0]):
            input_row = data[n]
            output_row = inverse_transformed_data[n]
            self.assertAlmostEqual(input_row[0], output_row[0], 5)


class TestTfidfEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.array([["Big yellow banana"], ["Big yellow bus"]])
        self.data_inf = np.array([["The sky is blue, a banana is yellow."]])

    def test_fit(self):
        # Test Case 1: Ensure whole vocabulary is learned when max_features is bigger then vocabulary
        encoder = encoders.TfidfEncoder(max_features=10)
        encoder.fit(self.data)
        self.assertCountEqual(
            encoder.encoder.get_feature_names_out(), ["banana", "big", "bus", "yellow"]
        )

        # Test Case 2: Ensure max features get's most frequent words
        encoder = encoders.TfidfEncoder(max_features=2)
        encoder.fit(self.data)
        self.assertCountEqual(
            encoder.encoder.get_feature_names_out(), ["big", "yellow"]
        )

    def test_transform(self):
        # Test Case 1: Verify that upon inference, it respects the vocab size of the training data
        encoder = encoders.TfidfEncoder(max_features=10)
        encoder.fit(self.data)
        encoded_features = encoder.transform(self.data_inf)
        self.assertEqual(encoded_features.values.shape, (1, 4))

        # Test Case 2: validate data type
        self.assertIsInstance(encoded_features, pd.DataFrame)


def test_get_column_transformer_from_features_iris(iris: IrisDataset):
    config = MLConfig(
        model_feature_id=iris.model_feature_id,
        problem_type=ProblemType.Classification,
        features=iris.features,
    )
    column_transformer = encoders.get_column_transformer(config, iris.X)

    transformer_ids = [t[0] for t in column_transformer.transformers]
    for feature in iris.features.values():
        if feature.id == iris.model_feature_id:
            continue
        assert feature.id in transformer_ids

    # difficult to assert on the transformer type so lets just check that the transform works
    Xt = column_transformer.fit_transform(iris.X)
    assert Xt is not None
    assert all(iris.X != Xt)


def test_get_column_transformer_from_dataframe_iris(iris: IrisDataset):
    config = MLConfig(
        model_feature_id=iris.model_feature_id,
        problem_type=ProblemType.Classification,
    )
    column_transformer = encoders.get_column_transformer(config, iris.X)

    transformer_ids = [t[0] for t in column_transformer.transformers]
    for feature in iris.features.values():
        if feature.id == iris.model_feature_id:
            continue
        assert feature.id in transformer_ids

    # difficult to assert on the transformer type so lets just check that the transform works
    Xt = column_transformer.fit_transform(iris.X)
    assert Xt is not None
    assert (iris.X != Xt).all(axis=None)


def test_get_model_feature_encoder_from_features_classification(iris: IrisDataset):
    y = iris.y.astype("category")
    y = iris.target_names[y]

    config = MLConfig(
        model_feature_id=iris.model_feature_id,
        problem_type=ProblemType.Classification,
        features=iris.features,
    )
    model_feature_encoder = encoders.get_model_feature_encoder(config, y)

    yt = model_feature_encoder.fit_transform(y)
    assert yt is not None
    assert y != yt


def test_get_model_feature_encoder_from_features_regression(diabetes: DiabetesDataset):

    config = MLConfig(
        model_feature_id=diabetes.model_feature_id,
        problem_type=ProblemType.Classification,
        features=diabetes.features,
    )
    model_feature_encoder = encoders.get_model_feature_encoder(config, diabetes.y)

    yt = model_feature_encoder.fit_transform(diabetes.y)
    assert yt is not None
    assert (diabetes.y != yt).all(axis=None)
