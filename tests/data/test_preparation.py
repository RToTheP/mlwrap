import os
import unittest

import pandas as pd
from mlwrap.data.cleaning import clean_rows

from mlwrap.data.preparation import (
    get_class_ratios,
    prepare_inference_data,
    prepare_training_data,
    split_data,
)
from mlwrap.config import MLConfig
from mlwrap.enums import DataType, Status
from tests.datasets import IrisDataset


class TestPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.iris = IrisDataset()

    def test_prepare_training_data_df_data_frame_missing(self):
        config = MLConfig()
        data_details = prepare_training_data(config, None)
        self.assertEqual(Status.invalid_data, data_details.status)

    def test_prepare_training_data_df(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        config = MLConfig(
            features=self.iris.features,
            model_feature_id=self.iris.model_feature_id,
        )

        # act
        data_details = prepare_training_data(config, df)

        # assert
        self.assertAlmostEqual(
            df.shape[0] * config.train_test_split, data_details.train_input.shape[0]
        )
        self.assertEqual(df.shape[1] - 1, data_details.train_input.shape[1])
        self.assertAlmostEqual(
            df.shape[0] * config.train_test_split, data_details.train_output.shape[0]
        )
        self.assertEqual(self.iris.target_count, data_details.train_output.shape[1])
        self.assertAlmostEqual(
            df.shape[0] * (1 - config.train_test_split),
            data_details.test_input.shape[0],
        )
        self.assertEqual(df.shape[1] - 1, data_details.test_input.shape[1])
        self.assertAlmostEqual(
            df.shape[0] * (1 - config.train_test_split),
            data_details.test_output.shape[0],
        )
        self.assertEqual(self.iris.target_count, data_details.test_output.shape[1])

    def test_prepare_inference_data_df(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        config = MLConfig(
            features=self.iris.features,
            model_feature_id=self.iris.model_feature_id,
        )
        # act
        training_data_details = prepare_training_data(config, df)
        config.encoder_bytes = training_data_details.get_encoder_bytes()
        data_details = prepare_inference_data(config, df)

        # assert
        self.assertEqual(
            df.shape[1],
            data_details.inference_input.shape[1],
            "Model feature should be removed",
        )

    def test_split_data(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        config = MLConfig(
            features=self.iris.features, model_feature_id=self.iris.model_feature_id
        )

        # act
        train, test = split_data(
            df,
            model_feature_id=config.model_feature_id,
            train_size=config.train_test_split,
            shuffle=config.shuffle_before_splitting,
            problem_type=config.problem_type,
        )
        # assert
        self.assertAlmostEqual(df.shape[0] * config.train_test_split, train.shape[0])
        self.assertAlmostEqual(
            df.shape[0] * (1 - config.train_test_split), test.shape[0]
        )

    def test_split_data_no_data(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        config = MLConfig(
            features=self.iris.features, model_feature_id=self.iris.model_feature_id
        )

        # act and assert
        self.assertRaises(
            ValueError,
            split_data,
            None,
            model_feature_id=config.model_feature_id,
            train_size=config.train_test_split,
            shuffle=config.shuffle_before_splitting,
            problem_type=config.problem_type,
        )

    def test_get_class_ratios_iris(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        config = MLConfig(
            features=self.iris.features, model_feature_id=self.iris.model_feature_id
        )

        n = 0
        for target_name in self.iris.target_names:
            print(
                f"Initial '{target_name}' value counts -",
                df["target"].value_counts()[n],
            )
            n += 1

        # clean rows - remove invalid rows
        _, df, _ = clean_rows(df, config)

        class_ratios_ = get_class_ratios(df, config)

        # assert
        one_third = 1.0 / 3.0
        self.assertTrue(all(cr == one_third for cr in class_ratios_.values()))
