import logging

import numpy as np
import pytest

from mlwrap import scores
from mlwrap.config import ExplanationResult
from mlwrap.enums import ProblemType

logging.getLogger().setLevel(logging.DEBUG)


def test_print_scores():
    # arrange
    scores_ = {"recall_macro": 0.5, "recall_weighted": 0.75}
    # act
    df = scores.print_scores(scores_)
    # assert
    assert df.shape[0] == 2
    assert df.shape[1] == 1


def test_print_explanation_result():
    # arrange
    explanation_result = ExplanationResult(
        feature_importances={"Colour": 0.5, "Temperature": 0.2}
    )
    # act
    df = scores.print_explanation_result(explanation_result)
    # assert
    assert df.shape[0] == 2
    assert df.shape[1] == 1


def test_get_scores_classification():
    y = np.array([1, 1, 0])
    y_pred = np.array([1, 1, 1])
    y_prob = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])

    scores_ = scores.get_scores(
        problem_type=ProblemType.Classification, y=y, y_pred=y_pred, y_prob=y_prob
    )

    assert pytest.approx(scores_["recall_weighted"]) == 2 / 3


def test_get_scores_regression():
    y = np.array([0.5, 1.0, 1.5, 0.2])
    y_pred = np.array([0.4, 1.2, 2, 0.2])

    scores_ = scores.get_scores(problem_type=ProblemType.Regression, y=y, y_pred=y_pred)
    assert pytest.approx(scores_["mean_abs_error"]) == 0.2
