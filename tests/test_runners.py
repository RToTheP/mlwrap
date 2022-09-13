import logging
import unittest
import pandas as pd

from mlwrap.config import MLConfig
from mlwrap.enums import AlgorithmType, ProblemType, ScoreType, Status
from mlwrap.runners import train, train_pipeline
from tests.datasets import DiabetesDataset, IrisDataset

logging.getLogger().setLevel(logging.DEBUG)


def test_train_lightgbm_decision_tree_regression(diabetes: DiabetesDataset):
    # arrange
    df = pd.concat([diabetes.X, diabetes.y], axis=1)

    config = MLConfig(
        algorithm_type=AlgorithmType.LightGBMDecisionTree,
        features=diabetes.features,
        model_feature_id=diabetes.model_feature_id,
    )

    # act
    result = train_pipeline(config=config, df=df)

    # assert
    assert result.model is not None
    assert result.scores[ScoreType.mean_absolute_error] < 70


def test_e2e_train_pipeline_sklearn_linear_classification(iris: IrisDataset):
    # arrange
    # Switch the target column to be string labels to test that scoring is working properly
    df = pd.concat([iris.X, iris.y], axis=1)
    df.target = iris.target_names[df.target]
    df.target = df.target.astype("category")

    config = MLConfig(
        algorithm_type=AlgorithmType.SklearnLinearModel,
        model_feature_id=iris.model_feature_id,
        problem_type=ProblemType.Classification,
        balance_data_via_resampling=True,
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
    predictions = result.model.predict(iris.X.head(n_inferences))

    # assert
    assert len(predictions) == n_inferences
    assert predictions[0] in iris.target_names


def test_e2e_lightgbm_decision_tree_classification(iris: IrisDataset):
    # arrange
    df = pd.concat([iris.X, iris.y], axis=1)

    config = MLConfig(
        algorithm_type=AlgorithmType.LightGBMDecisionTree,
        features=iris.features,
        model_feature_id=iris.model_feature_id,
    )

    # training
    # act
    training_results = train_pipeline(config=config, df=df)

    # assert
    assert training_results.model is not None
    assert training_results.scores[ScoreType.recall_weighted] > 0.8

    # inference
    # act
    n_inferences = 10
    X_test = iris.X.head(n_inferences)
    probabilities = training_results.model.predict_proba(X_test)
    predictions = training_results.model.predict(X_test)

    # assert
    assert len(predictions) == n_inferences
    assert iris.target_count == len(probabilities[0])
    assert predictions[0] in iris.y.unique()


def test_e2e_keras_neural_network_classification(iris):
    # arrange
    df = pd.concat([iris.X, iris.y], axis=1)
    df.target = iris.target_names[df.target]
    df.target = df.target.astype("category")

    config = MLConfig(
        algorithm_type=AlgorithmType.KerasNeuralNetwork,
        features=iris.features,
        model_feature_id=iris.model_feature_id,
    )

    # act
    results = train_pipeline(config=config, df=df)

    # assert
    assert results.model is not None
    assert results.scores[ScoreType.recall_weighted] > 0.7

    # inference
    # act
    n_inferences = 10
    probabilities = results.model.predict_proba(iris.X.head(n_inferences))
    predictions = results.model.predict(iris.X.head(n_inferences))

    # assert
    assert len(predictions) == n_inferences
    assert iris.target_count == len(probabilities[0])
    assert predictions[0] in iris.target_names


class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.iris = IrisDataset()
        cls.diabetes = DiabetesDataset()

    def test_xtrain_lightgbm_decision_tree_classification(self):
        # arrange
        df = pd.concat([self.iris.X, self.iris.y], axis=1)

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

    def test_xtrain_lightgbm_decision_tree_regression(self):
        # arrange
        df = pd.concat([self.diabetes.X, self.diabetes.y], axis=1)

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
