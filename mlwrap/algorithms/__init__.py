from mlwrap.algorithms.keras import get_keras
from mlwrap.algorithms.lightgbm import get_lightgbm
from mlwrap.algorithms.sklearn import (
    get_sklearn_linear_model,
    get_sklearn_decision_tree,
)
from mlwrap.enums import ProblemType


def get_default_algorithm(problem_type: ProblemType):
    return get_lightgbm(problem_type)

def get_iterations(model):
    # keras
    if hasattr(model, 'current_epoch'):
        return model.current_epoch
    
    # sklearn logistic regression
    if hasattr(model, 'n_iter_'):
        return model.n_iter_.max()

    return 1
