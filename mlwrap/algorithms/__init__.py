from mlwrap.algorithms.base import AlgorithmBase
from mlwrap.algorithms.sklearn import SklearnDecisionTree, SklearnLinearModel
from mlwrap.config import MLConfig
from mlwrap.enums import AlgorithmType


def get_algorithm(
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
