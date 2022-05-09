import logging
import unittest

import numpy as np

from mlwrap.config import (
    ExplanationResult,
    Feature,
    MLConfig,
)
from mlwrap.enums import FeatureType, ScoreType
from mlwrap.scores import get_scores, print_explanation_result, print_scores


class TestScores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger().setLevel(logging.DEBUG)

    def test_print_scores(self):
        # arrange
        scores = {ScoreType.recall_macro: 0.5, ScoreType.recall_weighted: 0.75}
        # act
        df = print_scores(scores)
        # assert
        self.assertEqual(2, df.shape[0])
        self.assertEqual(1, df.shape[1])

    def test_print_explanation_result(self):
        # arrange
        explanation_result = ExplanationResult(
            feature_importances = { "Colour" : 0.5, "Temperature" : 0.2 }
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
        predictions = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        actuals = np.array([[0, 1], [0, 1], [1, 0]])

        scores = get_scores(
            config=config,
            predict_fn=lambda x: predictions,
            inputs=None,
            actuals=actuals,
            encoders=None,
        )

        self.assertAlmostEqual(2 / 3, scores[ScoreType.recall_weighted])

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
        self.assertAlmostEqual(scores[ScoreType.mean_absolute_error], 0.2)
