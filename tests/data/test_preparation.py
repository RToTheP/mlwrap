import os
import unittest

import pandas as pd
from sklearn.datasets import load_iris

from mlwrap.data.preparation import prepare_data
from mlwrap.dto import Feature, InputData, MLSettings
from mlwrap.enums import DataType, FeatureType


class TestPreparation(unittest.TestCase):
    tmp_path_iris = "/tmp/iris.csv"
    iris = load_iris(as_frame=True)
    df_X = iris["data"]
    df_y = iris["target"]
    iris_model_feature_id = "target"
    iris_features = [*[
        Feature(id=name, feature_type=FeatureType.Continuous)
        for name in iris["feature_names"]
    ],Feature(id=iris_model_feature_id, feature_type=FeatureType.Categorical)]


    def setUp(self):
        if os.path.exists(self.tmp_path_iris):
            os.remove(self.tmp_path_iris)

    def test_prepare_data_csv_data_path_missing(self):

        settings = MLSettings(input_data=InputData(data_type=DataType.Csv))
        data_details = prepare_data(settings=settings)
        self.assertIsNone(data_details)

    def test_prepare_data_csv(self):
        # arrange
        df = pd.concat([self.df_X, self.df_y], axis=1)
        df.to_csv(self.tmp_path_iris, index=False)

        settings = MLSettings(
            features=self.iris_features,
            model_feature_id=self.iris_model_feature_id,
            input_data=InputData(data_type=DataType.Csv, data_path=self.tmp_path_iris),
        )

        # act
        data_details = prepare_data(settings=settings)

        # assert
        self.assertIsNotNone(data_details)

    def test_prepare_data_df_data_frame_missing(self):
        settings = MLSettings(input_data=InputData(data_type=DataType.DataFrame))
        data_details = prepare_data(settings)
        self.assertIsNone(data_details)

    def test_prepare_data_df(self):
        # arrange
        df = pd.concat([self.df_X, self.df_y], axis=1)

        settings = MLSettings(
            features=self.iris_features,
            model_feature_id=self.iris_model_feature_id,
            input_data=InputData(data_type=DataType.DataFrame, data_frame=df),
        )

        data_details = prepare_data(settings)
        self.assertIsNotNone(data_details)
