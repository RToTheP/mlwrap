import os
import unittest

import pandas as pd

from mlwrap.data.preparation import prepare_data, split_data
from mlwrap.dto import InputData, MLSettings
from mlwrap.enums import DataType
from tests.datasets import IrisDataset


class TestPreparation(unittest.TestCase):
    tmp_path_iris = "/tmp/iris.csv"

    @classmethod
    def setUpClass(cls) -> None:
        cls.iris = IrisDataset()

    def setUp(self):
        if os.path.exists(self.tmp_path_iris):
            os.remove(self.tmp_path_iris)

    def test_prepare_data_csv_data_path_missing(self):

        settings = MLSettings(input_data=InputData(data_type=DataType.Csv))
        data_details = prepare_data(settings=settings)
        self.assertIsNone(data_details)

    def test_prepare_data_csv(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)
        df.to_csv(self.tmp_path_iris, index=False)

        settings = MLSettings(
            features=self.iris.features,
            model_feature_id=self.iris.model_feature_id,
            input_data=InputData(data_type=DataType.Csv, data_path=self.tmp_path_iris),
        )

        # act
        data_details = prepare_data(settings=settings)

        # assert
        self.assertAlmostEqual(
            df.shape[0] * settings.train_test_split, data_details.train_input.shape[0]
        )
        self.assertEqual(df.shape[1] - 1, data_details.train_input.shape[1])
        self.assertAlmostEqual(
            df.shape[0] * settings.train_test_split, data_details.train_output.shape[0]
        )
        self.assertEqual(self.iris.target_count, data_details.train_output.shape[1])
        self.assertAlmostEqual(
            df.shape[0] * (1 - settings.train_test_split),
            data_details.test_input.shape[0],
        )
        self.assertEqual(df.shape[1] - 1, data_details.test_input.shape[1])
        self.assertAlmostEqual(
            df.shape[0] * (1 - settings.train_test_split),
            data_details.test_output.shape[0],
        )
        self.assertEqual(self.iris.target_count, data_details.test_output.shape[1])

    def test_prepare_data_df_data_frame_missing(self):
        settings = MLSettings(input_data=InputData(data_type=DataType.DataFrame))
        data_details = prepare_data(settings)
        self.assertIsNone(data_details)

    def test_prepare_data_df(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        settings = MLSettings(
            features=self.iris.features,
            model_feature_id=self.iris.model_feature_id,
            input_data=InputData(data_type=DataType.DataFrame, data_frame=df),
        )

        # act
        data_details = prepare_data(settings)

        # assert
        self.assertAlmostEqual(
            df.shape[0] * settings.train_test_split, data_details.train_input.shape[0]
        )
        self.assertEqual(df.shape[1] - 1, data_details.train_input.shape[1])
        self.assertAlmostEqual(
            df.shape[0] * settings.train_test_split, data_details.train_output.shape[0]
        )
        self.assertEqual(self.iris.target_count, data_details.train_output.shape[1])
        self.assertAlmostEqual(
            df.shape[0] * (1 - settings.train_test_split),
            data_details.test_input.shape[0],
        )
        self.assertEqual(df.shape[1] - 1, data_details.test_input.shape[1])
        self.assertAlmostEqual(
            df.shape[0] * (1 - settings.train_test_split),
            data_details.test_output.shape[0],
        )
        self.assertEqual(self.iris.target_count, data_details.test_output.shape[1])

    def test_split_data(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        settings = MLSettings(
            features=self.iris.features, model_feature_id=self.iris.model_feature_id
        )

        # act
        train, test = split_data(
            df,
            model_feature_id=settings.model_feature_id,
            train_size=settings.train_test_split,
            shuffle=settings.shuffle_before_splitting,
            problem_type=settings.get_problem_type(),
        )
        # assert
        self.assertAlmostEqual(df.shape[0] * settings.train_test_split, train.shape[0])
        self.assertAlmostEqual(
            df.shape[0] * (1 - settings.train_test_split), test.shape[0]
        )

    def test_split_data_no_data(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        settings = MLSettings(
            features=self.iris.features, model_feature_id=self.iris.model_feature_id
        )

        # act and assert
        self.assertRaises(
            ValueError,
            split_data,
            None,
            model_feature_id=settings.model_feature_id,
            train_size=settings.train_test_split,
            shuffle=settings.shuffle_before_splitting,
            problem_type=settings.get_problem_type(),
        )
