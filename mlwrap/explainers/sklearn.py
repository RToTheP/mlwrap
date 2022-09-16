from math import exp
from typing import List, Type

from mlwrap import utils
from mlwrap.enums import ProblemType
from mlwrap.config import ExplanationResult
from mlwrap.explainers.base import get_feature_importances, ExplainerBase


class SklearnLinearModelExplainer(ExplainerBase):
    def fit(self, X) -> ExplanationResult:
        # get the coefficients - this will differ depending on whether it is classification or regression
        coefficients = self._get_coefficients()

        # sum the coefficients for the features using the encoded feature indices
        feature_importances = get_feature_importances(
            self._column_transformer, importances=coefficients
        )
        return ExplanationResult(feature_importances=feature_importances)

    def explain(self, X) -> List[Type[ExplanationResult]]:
        # simple approach of multiplying the coefficients by the input and then partially summing
        coefficients = self._get_coefficients()
        explanation_results: List[Type[ExplanationResult]] = list()
        for input_ in X:
            importances = input_ * coefficients
            explanation_results.append(
                ExplanationResult(
                    feature_importances=get_feature_importances(
                        self._column_transformer, importances=importances
                    )
                )
            )
        return explanation_results

    def _get_coefficients(self):
        # get the coefficients - this will differ depending on whether it is classification or regression
        if self._config.problem_type == ProblemType.Classification:
            return self._model.coef_[0]
        elif self._config.problem_type == ProblemType.Regression:
            return self._model.coef_
        else:
            raise NotImplementedError


class SklearnDecisionTreeExplainer(ExplainerBase):
    def fit(self, X) -> ExplanationResult:
        # sum the coefficients for the features using the encoded feature indices
        feature_importances = get_feature_importances(
            self._column_transformer,
            importances=self._model.feature_importances_,
        )
        return ExplanationResult(feature_importances=feature_importances)

    def explain(self, X) -> List[Type[ExplanationResult]]:
        # simple approach of multiplying the coefficients by the input and then partially summing
        importances = X[:] * self._model.feature_importances_
        importances = utils.to_numpy(importances)
        explanation_results = [ExplanationResult(
                    feature_importances=get_feature_importances(
                        self._column_transformer, importances=importances[n]
                    )
                ) for n in range(importances.shape[0])]
        return explanation_results
