import abc
from typing import List, Type, Union

import numpy as np
import shap

from mlwrap.config import ExplanationResult, FeatureImportance
from mlwrap.data.config import DataDetails
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
        self, data_details: DataDetails
    ) -> Union[shap.GradientExplainer, shap.explainers.Tree, shap.explainers.Linear]:
        raise NotImplementedError

    def fit(self, data_details: DataDetails) -> ExplanationResult:
        explainer = self._get_explainer(data_details)

        shap_values = explainer.shap_values(data_details.background_data)

        # for training explanations we normalize globally
        importances = np.abs(shap_values)
        if len(importances.shape) == 3:
            importances = np.average(importances, axis=0)
        importances = np.average(importances, axis=0)

        feature_importances: List[Type[FeatureImportance]] = get_feature_importances(
            data_details=data_details, importances=importances
        )

        return ExplanationResult(feature_importances=feature_importances)

    def explain(self, data_details: DataDetails) -> List[Type[ExplanationResult]]:
        explainer = self._get_explainer(data_details)

        shap_values = explainer.shap_values(data_details.inference_input)

        # for inferences we normalize locally (i.e. per row)
        importances = np.abs(shap_values)
        if len(importances.shape) == 3:
            importances = np.average(importances, axis=0)

        return [
            ExplanationResult(
                feature_importances=get_feature_importances(
                    data_details=data_details, importances=x
                )
            )
            for x in importances
        ]


class GradientSHAP(SHAPExplainerBase):
    def _get_explainer(self, data_details: DataDetails) -> shap.GradientExplainer:
        explainer: shap.GradientExplainer = shap.GradientExplainer(
            self._algorithm._model, data_details.background_data
        )
        return explainer


class LinearSHAP(SHAPExplainerBase):
    def _get_explainer(self, data_details: DataDetails) -> shap.explainers.Linear:
        masker: shap.maskers.Independent = shap.maskers.Independent(
            data_details.background_data, self._config.explanation_background_samples
        )
        explainer: shap.explainers.Linear = shap.explainers.Linear(
            model=self._algorithm._model, masker=masker
        )
        return explainer


class TreeSHAP(SHAPExplainerBase):
    def _get_explainer(self, data_details: DataDetails) -> shap.explainers.Tree:
        explainer: shap.explainers.Tree = shap.explainers.Tree(
            model=self._algorithm._model
        )
        return explainer
