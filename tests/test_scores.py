import logging
import unittest

import numpy as np

from mlwrap.config import (
    ExplanationResult,
    Feature,
    FeatureImportance,
    MLConfig,
    ModelScore,
)
from mlwrap.enums import FeatureType, ScoreType
from mlwrap.scores import get_scores, print_explanation_result, print_scores


class TestScores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger().setLevel(logging.DEBUG)

    def test_print_scores_single_value(self):
        # arrange
        scores = [
            ModelScore(id=ScoreType.recall_macro, value=0.5),
            ModelScore(id=ScoreType.recall_weighted, value=0.75),
        ]
        # act
        df = print_scores(scores)
        # assert
        self.assertEqual(2, df.shape[0])
        self.assertEqual(1, df.shape[1])

    def test_print_scores_multiple_values(self):
        # arrange
        scores = [
            ModelScore(id=ScoreType.recall_macro, value=0.5, values=[0.25, 0.5, 0.75]),
            ModelScore(id=ScoreType.recall_weighted, value=0.9, values=[0.8, 0.9, 1.0]),
        ]
        # act
        df = print_scores(scores)

        # assert
        self.assertEqual(2, df.shape[0])
        self.assertEqual(6, df.shape[1])

    def test_print_explanation_result(self):
        # arrange
        explanation_result = ExplanationResult(
            feature_importances=[
                FeatureImportance(feature_id="Colour", value=0.5),
                FeatureImportance(feature_id="Temperature", value=0.2),
            ]
        )
        # act
        df = print_explanation_result(explanation_result)
        # assert
        self.assertEqual(2, df.shape[0])
        self.assertEqual(1, df.shape[1])

    def test_get_scores_classification(self):
        config = MLConfig(
            features=[
                Feature(id="a", feature_type=FeatureType.Categorical),
                Feature(id="b", feature_type=FeatureType.Categorical),
            ],
            model_feature_id="a",
        )
        predictions = np.array([[0.1,0.9], [0.2,0.8], [0.3,0.7]])
        actuals = np.array([[0,1], [0,1], [1,0]])

        scores = get_scores(
            config=config,
            predict_fn=lambda x: predictions,
            inputs=None,
            actuals=actuals,
            encoders=None,
        )

        recall_weighted = [s.value for s in scores if s.id == ScoreType.recall_weighted][0]

        self.assertAlmostEqual(2/3, recall_weighted)

    def test_get_scores_regression(self):
        config = MLConfig(
            features=[
                Feature(id="a", feature_type=FeatureType.Continuous),
                Feature(id="b", feature_type=FeatureType.Categorical),
            ],
            model_feature_id="a",
        )
        predictions = np.array([0.4, 1.2, 2, 0.2])
        actuals = np.array([0.5, 1.0, 1.5, 0.2])
        def inverse_transform(x):
            return x

        encoder = type(
            "DummyEncoder", (object,), {"inverse_transform": inverse_transform}
        )
        encoders = {"a": encoder}
        scores = get_scores(
            config=config,
            predict_fn=lambda x: predictions,
            inputs=None,
            actuals=actuals,
            encoders=encoders,
        )
        dct_scores = {s.id.name: s.value for s in scores}
        self.assertAlmostEqual(dct_scores[ScoreType.mean_absolute_error], 0.2)
