from typing import Any, Dict, Tuple

from mlwrap.enums import (
    AlgorithmType,
    ExplainerType,
    ProblemType,
)


class MLConfig:
    def __init__(
        self,
        algorithm_type: AlgorithmType = None,
        model_feature_id: str = None,
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
        problem_type: ProblemType = None,
        encoders: Dict[str, Any] = None
    ) -> None:
        self.algorithm_type = (
            algorithm_type
            if algorithm_type is not None
            else AlgorithmType.LightGBMDecisionTree
        )
        self.model_feature_id = model_feature_id
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
        self.problem_type = problem_type
        self.encoders = encoders


class ExplanationResult:
    def __init__(
        self,
        feature_importances: Dict[str, float] = None,
        feature_interactions: Dict[Tuple[str, str], float] = None,
    ):
        self.feature_importances = feature_importances
        self.feature_interactions = feature_interactions


class TrainingResults:
    def __init__(
        self,
        scores: Dict[str, float],
        model: Any,
        explanation_result: ExplanationResult = None,
    ) -> None:
        self.scores = scores if scores is not None else {}
        self.model = model
        self.explanation_result = explanation_result
