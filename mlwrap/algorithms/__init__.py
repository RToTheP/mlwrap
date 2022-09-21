from mlwrap.algorithms.keras import get_keras
from mlwrap.algorithms.lightgbm import get_lightgbm
from mlwrap.algorithms.sklearn import (
    get_sklearn_hist_grad_boosting_tree,
    get_sklearn_linear_model,
    get_sklearn_decision_tree,
)
from mlwrap.enums import ProblemType


def get_default_algorithm(problem_type: ProblemType):
    return get_sklearn_hist_grad_boosting_tree(problem_type)

def get_iterations(model):
    # keras
    if hasattr(model, 'current_epoch'):
        return model.current_epoch
    
    # sklearn 
    if hasattr(model, 'n_iter_'):
        n_iter = getattr(model, 'n_iter_')
        # some algorithms return an int
        if isinstance(n_iter, int):
            return n_iter
        # logistic regression returns an array of iterations
        return model.n_iter_.max()

    return 1
