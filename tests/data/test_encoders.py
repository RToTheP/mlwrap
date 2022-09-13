import numpy as np
import pandas as pd
import unittest

from mlwrap.config import MLConfig
from mlwrap.data.encoders import (
    CyclicalEncoder,
    TfidfEncoder,
    get_fitted_encoders,
    transform,
)
from tests.datasets import IrisDataset


class TestCyclicalEncoder(unittest.TestCase):
    def test_inverse_transform(self):
        # arrange
        cyclical_period = 7
        data = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        encoder = CyclicalEncoder(cyclical_period)

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
        encoder = TfidfEncoder(max_features=10)
        encoder.fit(self.data)
        self.assertCountEqual(
            encoder.encoder.get_feature_names(), ["banana", "big", "bus", "yellow"]
        )

        # Test Case 2: Ensure max features get's most frequent words
        encoder = TfidfEncoder(max_features=2)
        encoder.fit(self.data)
        self.assertCountEqual(encoder.encoder.get_feature_names(), ["big", "yellow"])

    def test_transform(self):
        # Test Case 1: Verify that upon inference, it respects the vocab size of the training data
        encoder = TfidfEncoder(max_features=10)
        encoder.fit(self.data)
        encoded_features = encoder.transform(self.data_inf)
        self.assertEqual(encoded_features.values.shape, (1, 4))

        # Test Case 2: validate data type
        self.assertIsInstance(encoded_features, pd.DataFrame)


class TestEncoders(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.iris = IrisDataset()

    def test_get_fitted_encoders(self):
        # arrange
        df = pd.concat([self.iris.X, self.iris.y], axis=1)

        config = MLConfig(
            features=self.iris.features, model_feature_id=self.iris.model_feature_id
        )

        # act
        encoders = get_fitted_encoders(df, config)

        # assert
        self.assertEqual(len(self.iris.features), len(encoders))
        self.assertTrue(all(f in encoders for f in self.iris.features))

    def test_get_fitted_encoders_no_data(self):
        # arrange
        config = MLConfig(
            features=self.iris.features, model_feature_id=self.iris.model_feature_id
        )

        # act and assert
        self.assertRaises(ValueError, get_fitted_encoders, None, config)

    def test_transform(self):
        # arrange
        df = pd.concat([self.iris.X, self.iris.y], axis=1)

        config = MLConfig(
            features=self.iris.features, model_feature_id=self.iris.model_feature_id
        )

        # act
        encoders = get_fitted_encoders(df, config)
        encoded_data, encoded_feature_indices = transform(df, config, encoders)

        # assert
        self.assertEqual(df.shape[0], encoded_data.shape[0])
        self.assertEqual(len(self.iris.features), len(encoded_feature_indices))
        total_indices = sum(efi.index_count for efi in encoded_feature_indices)
        self.assertEqual(total_indices, encoded_data.shape[1])

    def test_transform_no_data(self):
        # arrange
        df = pd.concat([self.iris.X, self.iris.y], axis=1)

        config = MLConfig(
            features=self.iris.features, model_feature_id=self.iris.model_feature_id
        )

        # act
        encoders = get_fitted_encoders(df, config)

        # assert
        self.assertRaises(ValueError, transform, None, config, encoders)
