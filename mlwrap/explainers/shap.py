import abc
from typing import List, Type, Union

import numpy as np
import shap

from mlwrap import sampling
from mlwrap.config import ExplanationResult
from mlwrap.explainers.base import get_feature_importances, ExplainerBase


class SHAPExplainerBase(ExplainerBase, metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "_get_explainer")
            and callable(subclass._get_explainer)
            or NotImplemented
        )

    @abc.abstractmethod
    def _get_explainer(
        self, background_data
    ) -> Union[shap.GradientExplainer, shap.explainers.Tree, shap.explainers.Linear]:
        raise NotImplementedError

    def fit(self, X, y) -> ExplanationResult:
        background_data = sampling.get_background_data(X, self._config)
        self._explainer = self._get_explainer(background_data)

        shap_values = self._explainer.shap_values(background_data)

        # for training explanations we normalize globally
        importances = np.abs(shap_values)
        if len(importances.shape) == 3:
            importances = np.average(importances, axis=0)
        importances = np.average(importances, axis=0)

        feature_importances = get_feature_importances(
            self._column_transformer, importances=importances
        )

        return ExplanationResult(feature_importances=feature_importances)

    def explain(self, X) -> List[Type[ExplanationResult]]:
        if self._explainer is None:
            raise ValueError("Explainer must be fitted")

        shap_values = self._explainer.shap_values(X)

        # for inferences we normalize locally (i.e. per row)
        importances = np.abs(shap_values)
        if len(importances.shape) == 3:
            importances = np.average(importances, axis=0)

        return [
            ExplanationResult(
                feature_importances=get_feature_importances(
                    self._column_transformer, importances=x
                )
            )
            for x in importances
        ]


class GradientSHAP(SHAPExplainerBase):
    def _get_explainer(self, background_data) -> shap.GradientExplainer:
        explainer: shap.GradientExplainer = shap.GradientExplainer(
            self._model, background_data
        )
        return explainer


class LinearSHAP(SHAPExplainerBase):
    def _get_explainer(self, background_data) -> shap.explainers.Linear:
        masker: shap.maskers.Independent = shap.maskers.Independent(
            background_data, self._config.explanation_background_samples
        )
        explainer: shap.explainers.Linear = shap.explainers.Linear(
            model=self._model, masker=masker
        )
        return explainer


class TreeSHAP(SHAPExplainerBase):
    def _get_explainer(self, background_data) -> shap.explainers.Tree:
        explainer: shap.explainers.Tree = shap.explainers.Tree(model=self._model)
        return explainer
