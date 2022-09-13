from mlwrap.config import MLConfig
from mlwrap.enums import ExplainerType, AlgorithmType

from mlwrap.explainers.base import ExplainerBase
from mlwrap.explainers.sklearn import (
    SklearnDecisionTreeExplainer,
    SklearnLinearModelExplainer,
)


def get_explainer(config: MLConfig, model, column_transformer):
    if config.explainer_type is not None:
        if config.explainer_type == ExplainerType.SklearnLinearModel:
            return SklearnLinearModelExplainer(config=config, model=model, column_transformer=column_transformer)
        elif config.explainer_type == ExplainerType.LightGBM:
            from mlwrap.explainers.lightgbm import LightGBMExplainer
            return LightGBMExplainer(config=config, model=model, column_transformer=column_transformer)

        elif config.explainer_type == ExplainerType.TreeSHAP:
            from mlwrap.explainers.shap import TreeSHAP
            return TreeSHAP(config=config, model=model, column_transformer=column_transformer)

        elif config.explainer_type == ExplainerType.GradientSHAP:
            from mlwrap.explainers.shap import GradientSHAP
            return GradientSHAP(config=config, model=model, column_transformer=column_transformer)

        elif config.explainer_type == ExplainerType.LinearSHAP:
            from mlwrap.explainers.shap import LinearSHAP
            return LinearSHAP(config=config, model=model, column_transformer=column_transformer)

        elif config.explainer_type == ExplainerType.SklearnDecisionTree:
            return SklearnDecisionTreeExplainer(config=config, model=model, column_transformer=column_transformer)

    
    if config.algorithm_type == AlgorithmType.KerasNeuralNetwork:
        from mlwrap.explainers.shap import GradientSHAP
        return GradientSHAP(config=config, model=model, column_transformer=column_transformer)

    elif config.algorithm_type in [
        AlgorithmType.LightGBMDecisionTree,
        AlgorithmType.LightGBMRandomForest,
    ]:
        from mlwrap.explainers.shap import TreeSHAP
        return TreeSHAP(config=config, model=model, column_transformer=column_transformer)

    elif config.algorithm_type == AlgorithmType.SklearnLinearModel:
        from mlwrap.explainers.shap import LinearSHAP
        return LinearSHAP(config=config, model=model, column_transformer=column_transformer)

    elif config.algorithm_type == AlgorithmType.SklearnDecisionTree:
        return SklearnDecisionTreeExplainer(config=config, model=model, column_transformer=column_transformer)

    raise NotImplementedError
