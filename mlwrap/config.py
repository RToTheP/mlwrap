from typing import Dict, List, Type, Union
import numpy as np

import pandas as pd

from mlwrap.enums import (
    AlgorithmType,
    CleaningType,
    DataType,
    EncoderType,
    ExplainerType,
    FeatureType,
    HandleUnknown,
    ProblemType,
    ScoreType,
    Status,
)


class Feature:
    def __init__(
        self,
        id: str,
        feature_type: FeatureType,
        active: bool = True,
        encoder_type: EncoderType = None,
        handle_unknown: HandleUnknown = None,
        default_value=None,
        min_value: float = None,
        max_value: float = None,
        cyclical_period: float = None,
        max_features=None,
        hash_size_ratio=None,
        model_min_value=None,
        model_max_value=None,
        model_labels: List[str] = None,
        allowed_labels: List[str] = None,
        other_label=None,
        keep_n_labels=None,
        label_percentage_threshold=None,
    ) -> None:
        self.id = id
        self.feature_type = feature_type
        self.active = active
        self.encoder_type = encoder_type
        self.handle_unknown = (
            handle_unknown if handle_unknown is not None else HandleUnknown.allow
        )
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.cyclical_period = cyclical_period
        self.max_features = max_features if max_features is not None else 10000
        self.hash_size_ratio = hash_size_ratio
        self.model_min_value = model_min_value
        self.model_max_value = model_max_value
        self.model_labels = model_labels
        self.allowed_labels = allowed_labels
        self.other_label = other_label if other_label is not None else "OTHER"
        self.keep_n_labels = max(keep_n_labels, 0) if keep_n_labels is not None else 10
        self.label_percentage_threshold = (
            max(0, min(label_percentage_threshold, 100))
            if label_percentage_threshold is not None
            else 10
        )


class InputData:
    def __init__(
        self,
        data_type: DataType,
        data_frame: pd.DataFrame = None,
        data_path: str = None,
    ) -> None:
        self.data_type = data_type
        self.data_frame = data_frame
        self.data_path = data_path


class MLConfig:
    def __init__(
        self,
        algorithm_type: AlgorithmType = None,
        features: List[Type[Feature]] = None,
        model_feature_id: str = None,
        input_data: InputData = None,
        train_test_split: float = 0.8,
        shuffle_before_splitting: bool = True,
        balance_data_via_resampling: bool = False,
        maximum_training_iterations: int = 1000,
        early_stopping_iterations: int = 50,
        adapt_class_weights: bool = False,
        model_training_batch_size: int = 32,
        maximum_tree_leaves: int = 31,
        maximum_tree_depth: int = -1,
        model_training_bagging_fraction: float = 0.5,
        model_training_bagging_frequency: int = 10,
        explain: bool = False,
        explainer_type: ExplainerType = None,
        explanation_background_samples: int = None,
        model_bytes: bytes = None,
        encoder_bytes: bytes = None,
        background_data_bytes: bytes = None,
    ) -> None:
        self.algorithm_type = (
            algorithm_type
            if algorithm_type is not None
            else AlgorithmType.LightGBMDecisionTree
        )
        self.features = features if features is not None else []
        self.model_feature_id = model_feature_id
        self.input_data = input_data
        self.train_test_split = train_test_split
        self.shuffle_before_splitting = shuffle_before_splitting
        self.balance_data_via_resampling = balance_data_via_resampling
        self.maximum_training_iterations = maximum_training_iterations
        self.early_stopping_iterations = early_stopping_iterations
        self.adapt_class_weights = adapt_class_weights
        self.model_training_batch_size = model_training_batch_size
        self.maximum_tree_leaves = maximum_tree_leaves
        self.maximum_tree_depth = maximum_tree_depth
        self.model_training_bagging_fraction = model_training_bagging_fraction
        self.model_training_bagging_frequency = model_training_bagging_frequency
        self.explain = explain
        self.explainer_type = explainer_type
        self.explanation_background_samples = min(
            1000,
            max(2, explanation_background_samples)
            if explanation_background_samples is not None
            else 100,
        )
        self.model_bytes = model_bytes
        self.encoder_bytes = encoder_bytes
        self.background_data_bytes = background_data_bytes

        self._problem_type = None
        self._model_feature = None

    @property
    def model_feature(self):
        if self._model_feature is None:
            self._model_feature = [
                f for f in self.features if f.id == self.model_feature_id
            ][0]
        return self._model_feature

    @property
    def problem_type(self):
        if self._problem_type is None:
            self._problem_type = (
                ProblemType.Classification
                if self.model_feature.feature_type == FeatureType.Categorical
                else ProblemType.Regression
            )
        return self._problem_type


class CleaningRecord:
    def __init__(
        self,
        row: Union[str, int] = None,
        label: str = None,
        value: float = None,
        feature: str = None,
        cleaning_type: CleaningType = None,
    ):
        self.row = row
        self.label = label
        self.value = value
        self.feature = feature
        self.cleaning_type = cleaning_type


class CleaningReport:
    def __init__(self):
        self.cleaning_records: List[CleaningRecord] = []

    def merge(self, other):
        self.cleaning_records = [*self.cleaning_records, *other.cleaning_records]

    def merge_cleaning_records(self, cleaning_records):
        self.cleaning_records = [*self.cleaning_records, *cleaning_records]


class FeatureInteraction:
    def __init__(
        self, feature_id1: str = None, feature_id2: str = None, value: float = None
    ):
        self.feature_id1 = feature_id1
        self.feature_id2 = feature_id2
        self.value = value


class ExplanationResult:
    def __init__(
        self,
        feature_importances: Dict[str, float] = None,
        feature_interactions: List[Type[FeatureInteraction]] = None,
    ):
        self.feature_importances = feature_importances
        self.feature_interactions = feature_interactions


class InferenceResult:
    def __init__(
        self,
        inference_id: str = None,
        feature_id: str = None,
        group_id: str = None,
        status: Status = None,
        values: List[str] = None,
        probabilities: List[float] = None,
        labels: List[str] = None,
        explanation_result: ExplanationResult = None,
    ):
        self.inference_id = inference_id
        self.feature_id = feature_id
        self.group_id = group_id
        self.status = status
        self.values = values
        self.probabilities = probabilities
        self.labels = labels
        self.explanation_result = explanation_result


class InferenceOutput:
    def __init__(
        self,
        status: Status = None,
        cleaning_report: CleaningReport = None,
        inference_results: List[Type[InferenceResult]] = None,
    ):
        self.status = status
        self.cleaning_report = cleaning_report
        self.inference_results = (
            inference_results if inference_results is not None else []
        )


class TrainingResults:
    def __init__(
        self,
        status: Status,
        scores: Dict[Type[ScoreType], float],
        cleaning_report: CleaningReport = None,
        model_bytes: bytes = None,
        encoder_bytes: bytes = None,
        explanation_result: ExplanationResult = None,
        background_data_bytes: bytes = None,
    ) -> None:
        self.status = status
        self.scores = scores if scores is not None else {}
        self.cleaning_report = cleaning_report
        self.model_bytes = model_bytes
        self.encoder_bytes = encoder_bytes
        self.explanation_result = explanation_result
        self.background_data_bytes = background_data_bytes
