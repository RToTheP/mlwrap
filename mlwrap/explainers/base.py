from __future__ import annotations

import abc
from typing import List, Type, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mlwrap.algorithms.base import AlgorithmBase

from mlwrap.data.config import DataDetails
from mlwrap.config import ExplanationResult, FeatureImportance, MLConfig


class ExplainerBase(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'fit') and
                callable(subclass.fit) and
                hasattr(subclass, 'explain') and
                callable(subclass.explain) or
                NotImplemented)

    @abc.abstractmethod
    def fit(self, data_details: DataDetails) -> ExplanationResult:
        raise NotImplementedError

    @abc.abstractmethod
    def explain(self, data_details: DataDetails) -> List[Type[ExplanationResult]]:
        raise NotImplementedError

    def __init__(
            self,
            config: MLConfig,
            algorithm: AlgorithmBase) -> None:
        self._config = config
        self._algorithm = algorithm


def get_feature_importances(data_details: DataDetails, importances: np.ndarray, normalize: bool = True):
    # sum the coefficients for the features using the encoded feature indices
    importances_ = [np.sum(importances[x.start_index: x.start_index + x.index_count])
                    for x in data_details.encoded_feature_indices]
    features = [x.feature_id for x in data_details.encoded_feature_indices]

    if normalize:
        importances_ = normalize_abs_values(importances_)

    feature_importances: List[Type[FeatureImportance]] = [FeatureImportance(
        feature_id=feature_id, value=importance) for feature_id, importance in zip(features, importances_)]

    return feature_importances


def normalize_abs_values(values):
    values = np.abs(values)
    min_ = min(values)
    max_ = max(values)
    denom_ = max_ - min_
    if denom_ == 0:
        denom_ = 1
    return (values - min_)/denom_
