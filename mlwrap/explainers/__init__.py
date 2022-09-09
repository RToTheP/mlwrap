from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlwrap.algorithms.base import AlgorithmBase
from mlwrap.config import MLConfig
from mlwrap.data.config import DataDetails
from mlwrap.enums import ExplainerType, AlgorithmType

from mlwrap.explainers.base import ExplainerBase
from mlwrap.explainers.sklearn import (
    SklearnDecisionTreeExplainer,
    SklearnLinearModelExplainer,
)


def get_explainer(config: MLConfig = None, algorithm: AlgorithmBase = None):
    if config.explainer_type is not None:
        if config.explainer_type == ExplainerType.SklearnLinearModel:
            return SklearnLinearModelExplainer(config=config, algorithm=algorithm)
        elif config.explainer_type == ExplainerType.LightGBM:
            from mlwrap.explainers.lightgbm import LightGBMExplainer

            return LightGBMExplainer(config=config, algorithm=algorithm)
        elif config.explainer_type == ExplainerType.TreeSHAP:
            from mlwrap.explainers.shap import TreeSHAP

            return TreeSHAP(config=config, algorithm=algorithm)
        elif config.explainer_type == ExplainerType.GradientSHAP:
            from mlwrap.explainers.shap import GradientSHAP

            return GradientSHAP(config=config, algorithm=algorithm)
            from mlwrap.explainers.shap import LinearSHAP
        elif config.explainer_type == ExplainerType.LinearSHAP:
            return LinearSHAP(config=config, algorithm=algorithm)
        elif config.explainer_type == ExplainerType.SklearnDecisionTree:
            return SklearnDecisionTreeExplainer(config=config, algorithm=algorithm)

    if algorithm is not None:
        if algorithm.algorithm == AlgorithmType.KerasNeuralNetwork:
            from mlwrap.explainers.shap import GradientSHAP

            return GradientSHAP(config=config, algorithm=algorithm)
        elif algorithm.algorithm in [
            AlgorithmType.LightGBMDecisionTree,
            AlgorithmType.LightGBMRandomForest,
        ]:
            from mlwrap.explainers.shap import TreeSHAP

            return TreeSHAP(config=config, algorithm=algorithm)
        elif algorithm.algorithm == AlgorithmType.SklearnLinearModel:
            from mlwrap.explainers.shap import LinearSHAP

            return LinearSHAP(config=config, algorithm=algorithm)
        elif algorithm.algorithm == AlgorithmType.SklearnDecisionTree:
            return SklearnDecisionTreeExplainer(config=config, algorithm=algorithm)

    raise NotImplementedError


def explain_model(
    config: MLConfig = None, algorithm=None, data_details: DataDetails = None
):
    explainer: ExplainerBase = get_explainer(config=config, algorithm=algorithm)
    return explainer.fit(data_details=data_details)
