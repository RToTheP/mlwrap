"""Module for Sklearn algorithms"""

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlwrap.config import MLConfig
from mlwrap.enums import ProblemType


def get_sklearn_decision_tree(config: MLConfig):
    if config.problem_type == ProblemType.Regression:
        return DecisionTreeRegressor()

    if config.problem_type == ProblemType.Classification:
        class_weight = "balanced" if config.adapt_class_weights else None
        return DecisionTreeClassifier(class_weight=class_weight)

    raise NotImplementedError


def get_sklearn_linear_model(config: MLConfig):
    if config.problem_type == ProblemType.Regression:
        return LinearRegression()

    if config.problem_type == ProblemType.Classification:
        class_weight = "balanced" if config.adapt_class_weights else None
        return LogisticRegression(
            max_iter=config.maximum_training_iterations,
            class_weight=class_weight,
        )
    else:
        raise NotImplementedError
