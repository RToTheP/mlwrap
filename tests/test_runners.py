import logging
import unittest
import pandas as pd
import pytest

from mlwrap.config import MLConfig
from mlwrap.enums import AlgorithmType, DataType, ProblemType, ScoreType, Status
from mlwrap.runners import infer, train, train_pipeline
from tests.datasets import DiabetesDataset, IrisDataset

logging.getLogger().setLevel(logging.DEBUG)

@pytest.fixture
def iris():
    return IrisDataset()

@pytest.fixture
def diabetes():
    return DiabetesDataset()

def test_e2e_train_pipeline_iris(iris):
    # arrange
    df = pd.concat([iris.df_X, iris.df_y], axis=1)

    config = MLConfig(
        algorithm_type=AlgorithmType.SklearnDecisionTree,
        model_feature_id=iris.model_feature_id,
        problem_type=ProblemType.Classification,
        balance_data_via_resampling= True
    )

    # training
    # act
    result = train_pipeline(config=config, df=df)

    # assert
    assert result is not None
    assert result.scores[ScoreType.recall_weighted] > 0.8

    # inference
    # act
    n_inferences = 10
    df = pd.concat([iris.df_X, iris.df_y], axis=1).head(n_inferences)
    df.pop(iris.model_feature_id)
    predictions = result.model.predict(df)

    # assert
    assert len(df) == len(predictions)


class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:        
        cls.iris = IrisDataset()
        cls.diabetes = DiabetesDataset()

    def test_e2e_lightgbm_decision_tree_classification(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        config = MLConfig(
            algorithm_type=AlgorithmType.LightGBMDecisionTree,
            features=self.iris.features,
            model_feature_id=self.iris.model_feature_id,
        )

        # training
        # act
        training_results = train(config=config, df=df)

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
        config.model_bytes = training_results.model_bytes
        config.encoder_bytes = training_results.encoder_bytes
        config.background_data_bytes = training_results.background_data_bytes

        inference_output = infer(config=config, df=df)

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
            explain=True,
        )

        # act
        result = train(config=config, df=df)

        # assert
        self.assertEqual(Status.success, result.status)
        self.assertIsNotNone(result.model_bytes)
        self.assertEqual(df.shape[0], result.scores[ScoreType.total_row_count])
        self.assertTrue(result.scores[ScoreType.recall_weighted] > 0.8)
        self.assertEqual(
            len(config.features) - 1, len(result.explanation_result.feature_importances)
        )
        self.assertTrue(
            any(
                value > 0
                for value in result.explanation_result.feature_importances.values()
            )
        )
        self.assertEqual(
            1, result.explanation_result.feature_importances["petal length (cm)"]
        )

    def test_train_lightgbm_decision_tree_regression(self):
        # arrange
        df = pd.concat([self.diabetes.df_X, self.diabetes.df_y], axis=1)

        config = MLConfig(
            algorithm_type=AlgorithmType.LightGBMDecisionTree,
            features=self.diabetes.features,
            model_feature_id=self.diabetes.model_feature_id,
        )

        # act
        result = train(config=config, df=df)

        # assert
        self.assertEqual(Status.success, result.status)
        self.assertIsNotNone(result.model_bytes)
        self.assertEqual(df.shape[0], result.scores[ScoreType.total_row_count])
        self.assertTrue(result.scores[ScoreType.mean_absolute_error] < 70)

    def test_xtrain_lightgbm_decision_tree_regression(self):
        # arrange
        df = pd.concat([self.diabetes.df_X, self.diabetes.df_y], axis=1)

        config = MLConfig(
            algorithm_type=AlgorithmType.LightGBMDecisionTree,
            features=self.diabetes.features,
            model_feature_id=self.diabetes.model_feature_id,
            explain=True,
        )

        # act
        result = train(config=config, df=df)

        # assert
        self.assertEqual(Status.success, result.status)
        self.assertIsNotNone(result.model_bytes)
        self.assertEqual(df.shape[0], result.scores[ScoreType.total_row_count])
        self.assertTrue(result.scores[ScoreType.mean_absolute_error] < 70)
        self.assertEqual(
            len(config.features) - 1, len(result.explanation_result.feature_importances)
        )
        self.assertTrue(
            any(
                value > 0
                for value in result.explanation_result.feature_importances.values()
            )
        )
        max_feature_importance = max(
            result.explanation_result.feature_importances,
            key=result.explanation_result.feature_importances.get,
        )
        self.assertTrue(max_feature_importance in ["s5", "bmi"])

    def test_train_keras_neural_network_classification(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        config = MLConfig(
            algorithm_type=AlgorithmType.KerasNeuralNetwork,
            features=self.iris.features,
            model_feature_id=self.iris.model_feature_id,
        )

        # act
        result = train(config=config, df=df)

        # assert
        self.assertEqual(Status.success, result.status)
        self.assertIsNotNone(result.model_bytes)
        self.assertEqual(df.shape[0], result.scores[ScoreType.total_row_count])
        self.assertTrue(result.scores[ScoreType.recall_weighted] > 0.8)
