"""Module for Sklearn algorithms"""

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlwrap.enums import ProblemType

max_iter = 1000

def get_sklearn_decision_tree(problem_type: ProblemType, adapt_class_weights: bool = False):
    if problem_type == ProblemType.Regression:
        return DecisionTreeRegressor()

    if problem_type == ProblemType.Classification:
        class_weight = "balanced" if adapt_class_weights else None
        return DecisionTreeClassifier(class_weight=class_weight)

    raise NotImplementedError


def get_sklearn_linear_model(problem_type: ProblemType, adapt_class_weights: bool = False):
    if problem_type == ProblemType.Regression:
        return LinearRegression()

    if problem_type == ProblemType.Classification:
        class_weight = "balanced" if adapt_class_weights else None
        return LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
        )
    
    raise NotImplementedError
