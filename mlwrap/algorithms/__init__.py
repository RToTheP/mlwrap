from mlwrap.algorithms.base import AlgorithmBase
from mlwrap.algorithms.sklearn import (
    SklearnDecisionTree,
    SklearnLinearModel,
    get_sklearn_linear_model,
    get_sklearn_decision_tree,
)
from mlwrap.config import MLConfig
from mlwrap.enums import AlgorithmType


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


def get_algorithm_old(
    name: AlgorithmType = None, config: MLConfig = None, stop_event=None
) -> AlgorithmBase:
    """Factory method to select algorithm class"""
    alg_name = name if name is not None else config.algorithm_type
    if alg_name == AlgorithmType.KerasNeuralNetwork:
        from mlwrap.algorithms.keras import KerasNeuralNetwork

        return KerasNeuralNetwork(config, stop_event)
    elif alg_name in [
        AlgorithmType.LightGBMDecisionTree,
        AlgorithmType.LightGBMRandomForest,
    ]:
        from mlwrap.algorithms.lightgbm import LightGBMWrapper

        return LightGBMWrapper(config, stop_event, alg_name)
    elif alg_name == AlgorithmType.SklearnLinearModel:
        return SklearnLinearModel(config, stop_event)
    elif alg_name == AlgorithmType.SklearnDecisionTree:
        return SklearnDecisionTree(config, stop_event)
    else:
        raise NotImplementedError
