import logging
import unittest
import pandas as pd

from mlwrap.config import InputData, MLConfig
from mlwrap.enums import AlgorithmType, DataType, ScoreType, Status
from mlwrap.runners import infer, train
from tests.datasets import DiabetesDataset, IrisDataset


class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        cls.iris = IrisDataset()
        cls.diabetes = DiabetesDataset()

    def test_e2e_lightgbm_decision_tree_classification(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        config = MLConfig(
            algorithm_type=AlgorithmType.LightGBMDecisionTree,
            features=self.iris.features,
            model_feature_id=self.iris.model_feature_id,
            input_data=InputData(data_type=DataType.DataFrame, data_frame=df),
        )

        # training
        # act
        training_results = train(config=config)

        # assert
        self.assertEqual(Status.success, training_results.status)
        self.assertIsNotNone(training_results.model_bytes)
        self.assertEqual(
            df.shape[0], training_results.scores[ScoreType.total_row_count]
        )
        self.assertTrue(training_results.scores[ScoreType.recall_weighted] > 0.8)

        # inference
        # act
        n_inferences = 10
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1).head(n_inferences)
        config.input_data = InputData(data_type=DataType.DataFrame, data_frame=df)
        config.model_bytes = training_results.model_bytes
        config.encoder_bytes = training_results.encoder_bytes
        config.background_data_bytes = training_results.background_data_bytes

        inference_output = infer(config=config)

        # assert
        self.assertEqual(n_inferences, len(inference_output.inference_results))
        self.assertEqual(
            self.iris.target_count,
            len(inference_output.inference_results[0].probabilities),
        )

    def test_xtrain_lightgbm_decision_tree_classification(self):
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
        self.assertIsNotNone(result.model_bytes)
        self.assertEqual(df.shape[0], result.scores[ScoreType.total_row_count])
        self.assertTrue(result.scores[ScoreType.recall_weighted] > 0.8)
        self.assertEqual(
            len(config.features) - 1, len(result.explanation_result.feature_importances)
        )
        self.assertTrue(
            any(fi.value > 0 for fi in result.explanation_result.feature_importances)
        )
        petal_length_importance = [
            fi.value
            for fi in result.explanation_result.feature_importances
            if fi.feature_id == "petal length (cm)"
        ][0]
        self.assertEqual(1, petal_length_importance)

    def test_train_lightgbm_decision_tree_regression(self):
        # arrange
        df = pd.concat([self.diabetes.df_X, self.diabetes.df_y], axis=1)

        config = MLConfig(
            algorithm_type=AlgorithmType.LightGBMDecisionTree,
            features=self.diabetes.features,
            model_feature_id=self.diabetes.model_feature_id,
            input_data=InputData(data_type=DataType.DataFrame, data_frame=df),
        )

        # act
        result = train(config=config)

        # assert
        self.assertEqual(Status.success, result.status)
        self.assertIsNotNone(result.model_bytes)
        self.assertEqual(df.shape[0], result.scores[ScoreType.total_row_count])
        self.assertTrue(result.scores[ScoreType.mean_absolute_error] < 60)

    def test_xtrain_lightgbm_decision_tree_regression(self):
        # arrange
        df = pd.concat([self.diabetes.df_X, self.diabetes.df_y], axis=1)

        config = MLConfig(
            algorithm_type=AlgorithmType.LightGBMDecisionTree,
            features=self.diabetes.features,
            model_feature_id=self.diabetes.model_feature_id,
            input_data=InputData(data_type=DataType.DataFrame, data_frame=df),
            explain=True,
        )

        # act
        result = train(config=config)

        # assert
        self.assertEqual(Status.success, result.status)
        self.assertIsNotNone(result.model_bytes)
        self.assertEqual(df.shape[0], result.scores[ScoreType.total_row_count])
        self.assertTrue(result.scores[ScoreType.mean_absolute_error] < 60)
        self.assertEqual(
            len(config.features) - 1, len(result.explanation_result.feature_importances)
        )
        self.assertTrue(
            any(fi.value > 0 for fi in result.explanation_result.feature_importances)
        )
        s5_importance = [
            fi.value
            for fi in result.explanation_result.feature_importances
            if fi.feature_id == "s5"
        ][0]
        self.assertEqual(1, s5_importance)

    def test_train_keras_neural_network_classification(self):
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
        self.assertIsNotNone(result.model_bytes)
        self.assertEqual(df.shape[0], result.scores[ScoreType.total_row_count])
        self.assertTrue(result.scores[ScoreType.recall_weighted] > 0.8)
