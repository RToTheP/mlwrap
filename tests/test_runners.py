import logging
import unittest
import pandas as pd

from mlwrap.config import InputData, MLConfig
from mlwrap.enums import AlgorithmType, DataType, ScoreType, Status
from mlwrap.runners import train
from tests.datasets import IrisDataset


class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        cls.iris = IrisDataset()

    def test_train_lightgbm_decision_tree(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        config = MLConfig(
            algorithm_type=AlgorithmType.LightGBMDecisionTree,
            features=self.iris.features,
            model_feature_id=self.iris.model_feature_id,
            input_data=InputData(data_type=DataType.DataFrame, data_frame=df),
        )

        # act
        result = train(config=config)

        # assert
        self.assertEqual(Status.success, result.status)
        self.assertIsNotNone(result.model_details)
        dct_scores = { s.id.name : s.value for s in result.scores}
        self.assertEqual(df.shape[0], dct_scores[ScoreType.total_row_count])
        
    def test_xtrain_lightgbm_decision_tree(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        config = MLConfig(
            algorithm_type=AlgorithmType.LightGBMDecisionTree,
            features=self.iris.features,
            model_feature_id=self.iris.model_feature_id,
            input_data=InputData(data_type=DataType.DataFrame, data_frame=df),
            explain=True,
        )

        # act
        result = train(config=config)

        # assert
        self.assertEqual(Status.success, result.status)

    def test_train_keras_neural_network(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        config = MLConfig(
            algorithm_type=AlgorithmType.KerasNeuralNetwork,
            features=self.iris.features,
            model_feature_id=self.iris.model_feature_id,
            input_data=InputData(data_type=DataType.DataFrame, data_frame=df),
        )

        # act
        result = train(config=config)

        # assert
        self.assertEqual(Status.success, result.status)
