from typing import List, Type

import numpy as np

from mlwrap.enums import ProblemType, AlgorithmType
from mlwrap.config import ExplanationResult, FeatureImportance
from mlwrap.data.config import DataDetails
from mlwrap.explainers.base import get_feature_importances, ExplainerBase


class SklearnLinearModelExplainer(ExplainerBase):
    def _is_valid(self):
        if self._algorithm.algorithm != AlgorithmType.SklearnLinearModel:
            raise NotImplementedError

    def fit(self, data_details: DataDetails) -> ExplanationResult:
        self._is_valid()

        # get the coefficients - this will differ depending on whether it is classification or regression
        coefficients = self._get_coefficients()

        # sum the coefficients for the features using the encoded feature indices
        feature_importances: List[Type[FeatureImportance]] = get_feature_importances(
            data_details=data_details, importances=coefficients)
        return ExplanationResult(feature_importances=feature_importances)

    def explain(self, data_details: DataDetails) -> List[Type[ExplanationResult]]:
        self._is_valid()

        # simple approach of multiplying the coefficients by the input and then partially summing
        coefficients = self._get_coefficients()
        explanation_results: List[Type[ExplanationResult]] = list()
        for input_ in data_details.inference_input:
            importances = input_ * coefficients
            explanation_results.append(ExplanationResult(feature_importances=get_feature_importances(
                data_details=data_details, importances=importances)))
        return explanation_results

    def _get_coefficients(self):
        # get the coefficients - this will differ depending on whether it is classification or regression
        if self._config.problem_type == ProblemType.Classification:
            return self._algorithm._model.coef_[0]
        elif self._config.problem_type == ProblemType.Regression:
            return self._algorithm._model.coef_
        else:
            raise NotImplementedError


class SklearnDecisionTreeExplainer(ExplainerBase):
    def _is_valid(self):
        if self._algorithm.algorithm != AlgorithmType.SklearnDecisionTree:
            raise NotImplementedError

    def fit(self, data_details: DataDetails) -> ExplanationResult:
        self._is_valid()

        # sum the coefficients for the features using the encoded feature indices
        feature_importances: List[Type[FeatureImportance]] = get_feature_importances(
            data_details=data_details, importances=self._algorithm._model.feature_importances_)
        return ExplanationResult(feature_importances=feature_importances)

    def explain(self, data_details: DataDetails) -> List[Type[ExplanationResult]]:
        self._is_valid()

        # simple approach of multiplying the coefficients by the input and then partially summing
        explanation_results: List[Type[ExplanationResult]] = list()
        for input_ in data_details.inference_input:
            importances = input_ * self._algorithm._model.feature_importances_
            explanation_results.append(ExplanationResult(feature_importances=get_feature_importances(
                data_details=data_details, importances=importances)))
        return explanation_results
