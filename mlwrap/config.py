from typing import Any, Dict, Tuple

from mlwrap.enums import ProblemType


class MLConfig:
    def __init__(
        self,
        model_feature_id: str = None,
        train_test_split: float = 0.8,
        shuffle_before_splitting: bool = True,
        balance_data_via_resampling: bool = False,
        explain: bool = False,
        explanation_background_samples: int = None,
        problem_type: ProblemType = None,
        algorithm: Any = None,
        encoders: Dict[str, Any] = None
    ) -> None:
        self.model_feature_id = model_feature_id
        self.train_test_split = train_test_split
        self.shuffle_before_splitting = shuffle_before_splitting
        self.balance_data_via_resampling = balance_data_via_resampling
        self.explain = explain
        self.explanation_background_samples = min(
            1000,
            max(2, explanation_background_samples)
            if explanation_background_samples is not None
            else 100,
        )
        self.problem_type = problem_type
        self.algorithm = algorithm
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
