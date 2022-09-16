import abc
from typing import Dict, List, Type

import numpy as np
from sklearn.compose import ColumnTransformer

from mlwrap.config import ExplanationResult, MLConfig


class ExplainerBase(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "fit")
            and callable(subclass.fit)
            and hasattr(subclass, "explain")
            and callable(subclass.explain)
            or NotImplemented
        )

    @abc.abstractmethod
    def fit(self, X) -> ExplanationResult:
        raise NotImplementedError

    @abc.abstractmethod
    def explain(self, X) -> List[Type[ExplanationResult]]:
        raise NotImplementedError

    def __init__(self, config: MLConfig, model, column_transformer, background_data) -> None:
        self._config = config
        self._model = model
        self._column_transformer = column_transformer
        self._background_data = background_data


def get_feature_importances(
    column_transformer: ColumnTransformer,
    importances: np.ndarray,
    normalize: bool = True,
) -> Dict[str, float]:
    encoded_feature_indices = column_transformer.output_indices_
    # sum the coefficients for the features using the encoded feature indices
    importances_ = [np.sum(importances[x]) for x in encoded_feature_indices.values()]
    features = [x for x in encoded_feature_indices]

    if normalize:
        importances_ = normalize_abs_values(importances_)

    feature_importances = {
        feature_id: importance for feature_id, importance in zip(features, importances_)
    }

    return feature_importances


def normalize_abs_values(values):
    values = np.abs(values)
    min_ = min(values)
    max_ = max(values)
    denom_ = max_ - min_
    if denom_ == 0:
        denom_ = 1
    return (values - min_) / denom_
