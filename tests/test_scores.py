from typing import List
import logging
import unittest

from mlwrap.config import ExplanationResult, FeatureImportance, ModelScore
from mlwrap.enums import ScoreType
from mlwrap.scores import print_explanation_result, print_scores


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
