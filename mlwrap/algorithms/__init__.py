from mlwrap.algorithms.sklearn import (
    get_sklearn_linear_model,
    get_sklearn_decision_tree,
)
from mlwrap.config import MLConfig
from mlwrap.enums import AlgorithmType, ProblemType


def get_algorithm(config: MLConfig, X_train, X_test, y_train, y_test):
    """Factory method to select algorithm"""
    if config.algorithm_type == AlgorithmType.KerasNeuralNetwork:
        from mlwrap.algorithms.keras import get_keras

        return get_keras(config, X_train, X_test, y_train, y_test)

    if config.algorithm_type in [
        AlgorithmType.LightGBMDecisionTree,
        AlgorithmType.LightGBMRandomForest,
    ]:
        from mlwrap.algorithms.lightgbm import get_lightgbm

        return get_lightgbm(config)
    elif config.algorithm_type == AlgorithmType.SklearnLinearModel:
        return get_sklearn_linear_model(config)
    elif config.algorithm_type == AlgorithmType.SklearnDecisionTree:
        return get_sklearn_decision_tree(config)
    else:
        raise NotImplementedError

def get_iterations(config: MLConfig, model):
    if config.algorithm_type == AlgorithmType.KerasNeuralNetwork:
        return model.current_epoch
    if config.algorithm_type in [
        AlgorithmType.LightGBMDecisionTree,
        AlgorithmType.LightGBMRandomForest,
    ]:
        return 1
    elif config.algorithm_type == AlgorithmType.SklearnLinearModel:
        if config.problem_type == ProblemType.Classification:
            return model.n_iter_.max()
        elif config.problem_type == ProblemType.Regression:
            return 1  # doesn't mean anything for linear regression
    elif config.algorithm_type == AlgorithmType.SklearnDecisionTree:
        return 1
    else:
        raise NotImplementedError
