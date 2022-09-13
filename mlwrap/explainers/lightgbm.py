from typing import List, Type


from mlwrap.config import ExplanationResult
from mlwrap.explainers.base import get_feature_importances, ExplainerBase


class LightGBMExplainer(ExplainerBase):
    def fit(self, X, y) -> ExplanationResult:
        # sum the coefficients for the features using the encoded feature indices
        feature_importances = get_feature_importances(
            self._column_transformer,
            importances=self._model.feature_importances_,
        )
        return ExplanationResult(feature_importances=feature_importances)

    def explain(self, X) -> List[Type[ExplanationResult]]:
        # simple approach of multiplying the coefficients by the input and then partially summing
        explanation_results: List[Type[ExplanationResult]] = list()
        for input_ in X:
            importances = input_ * self._model.feature_importances_
            explanation_results.append(
                ExplanationResult(
                    feature_importances=get_feature_importances(
                        self._column_transformer, importances=importances
                    )
                )
            )
        return explanation_results
