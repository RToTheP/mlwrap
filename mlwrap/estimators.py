import logging

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 

from mlwrap.config import MLConfig
from mlwrap.enums import AlgorithmType, ProblemType


def _get_sklearn_decision_tree(config: MLConfig):
    logging.info(f"{config.problem_type} model")
    if config.problem_type == ProblemType.Regression:
        return DecisionTreeRegressor()
    elif config.problem_type == ProblemType.Classification:
        class_weight = "balanced" if config.adapt_class_weights else None
        return DecisionTreeClassifier(class_weight=class_weight)
    else:
        raise NotImplementedError

def _get_sklearn_linear_model(config: MLConfig):  
    logging.info(f"{config.problem_type} model")
    if config.problem_type == ProblemType.Regression:
        return LinearRegression()
    elif config.problem_type == ProblemType.Classification:
        class_weight = "balanced" if config.adapt_class_weights else None
        return LogisticRegression(
            max_iter=config.maximum_training_iterations,
            class_weight=class_weight,
        )
    else:
        raise NotImplementedError

def get_estimator(name: AlgorithmType = None, config: MLConfig = None):
    """Factory method to select algorithm class"""
    alg_name = name if name is not None else config.algorithm_type
    # if alg_name == AlgorithmType.KerasNeuralNetwork:
    #     from mlwrap.algorithms.keras import KerasNeuralNetwork

    #     return KerasNeuralNetwork(config, stop_event)
    # elif alg_name in [
    #     AlgorithmType.LightGBMDecisionTree,
    #     AlgorithmType.LightGBMRandomForest,
    # ]:
    #     from mlwrap.algorithms.lightgbm import LightGBMWrapper

    #     return LightGBMWrapper(config, stop_event, alg_name)
    if alg_name == AlgorithmType.SklearnLinearModel:
        return _get_sklearn_linear_model(config)
    elif alg_name == AlgorithmType.SklearnDecisionTree:
        return _get_sklearn_decision_tree(config)
    else:
        raise NotImplementedError