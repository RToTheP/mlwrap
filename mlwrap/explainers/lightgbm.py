from typing import List, Type

from mlwrap.enums import AlgorithmType
from mlwrap.config import ExplanationResult
from mlwrap.data.config import DataDetails
from mlwrap.explainers.base import get_feature_importances, ExplainerBase


class LightGBMExplainer(ExplainerBase):
    def _is_valid(self) -> None:
        if self._algorithm.algorithm not in [
            AlgorithmType.LightGBMDecisionTree,
            AlgorithmType.LightGBMRandomForest,
        ]:
            raise NotImplementedError

    def fit(self, data_details: DataDetails) -> ExplanationResult:
        self._is_valid()

        # sum the coefficients for the features using the encoded feature indices
        feature_importances = get_feature_importances(
            data_details=data_details,
            importances=self._algorithm._model.feature_importances_,
        )
        return ExplanationResult(feature_importances=feature_importances)

    def explain(self, data_details: DataDetails) -> List[Type[ExplanationResult]]:
        self._is_valid()

        # simple approach of multiplying the coefficients by the input and then partially summing
        explanation_results: List[Type[ExplanationResult]] = list()
        for input_ in data_details.inference_input:
            importances = input_ * self._algorithm._model.feature_importances_
            explanation_results.append(
                ExplanationResult(
                    feature_importances=get_feature_importances(
                        data_details=data_details, importances=importances
                    )
                )
            )
        return explanation_results
