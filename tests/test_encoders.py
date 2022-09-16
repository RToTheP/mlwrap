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


def test_get_column_transformer_iris(iris: IrisDataset):
    column_transformer = encoders.get_column_transformer(MLConfig(), iris.X)

    transformer_ids = [t[0] for t in column_transformer.transformers]
    assert all(transformer_ids == iris.X.columns)

    # difficult to assert on the transformer type so lets just check that the transform works
    Xt = column_transformer.fit_transform(iris.X)
    assert Xt is not None
    assert (iris.X != Xt).all(axis=None)

def test_get_column_transformer_with_override_iris(iris: IrisDataset):
    encoders_ = {'petal length (cm)' : encoders.FeatureHasherWrapper() }
    column_transformer = encoders.get_column_transformer(MLConfig(encoders=encoders_), iris.X)

    transformer_ids = [t[0] for t in column_transformer.transformers]
    assert all(transformer_ids == iris.X.columns)

    # check that the override works
    for t in column_transformer.transformers:
        if t[0] == 'petal length (cm)':
            assert isinstance(t[1].named_steps['encoder'], encoders.FeatureHasherWrapper)

def test_get_model_feature_encoder_classification(iris: IrisDataset):

    config = MLConfig(
        model_feature_id=iris.model_feature_id,
        problem_type=ProblemType.Classification,
    )
    model_feature_encoder = encoders.get_model_feature_encoder(config, iris.y)

    yt = model_feature_encoder.fit_transform(iris.y)
    assert yt is not None
    assert (iris.y == yt).all(axis=None)

def test_get_model_feature_encoder_with_overrides_classification(iris: IrisDataset):
    encoders_ = {'target' : encoders.FeatureHasherWrapper() }
    config = MLConfig(
        model_feature_id=iris.model_feature_id,
        problem_type=ProblemType.Classification,
        encoders=encoders_
    )
    model_feature_encoder = encoders.get_model_feature_encoder(config, iris.y)

    assert isinstance(model_feature_encoder.named_steps['encoder'], encoders.FeatureHasherWrapper)


def test_get_model_feature_encoder_regression(diabetes: DiabetesDataset):

    config = MLConfig(
        model_feature_id=diabetes.model_feature_id,
        problem_type=ProblemType.Regression,
    )
    model_feature_encoder = encoders.get_model_feature_encoder(config, diabetes.y)

    yt = model_feature_encoder.fit_transform(diabetes.y)
    assert yt is not None
    assert (diabetes.y != yt).all(axis=None)
