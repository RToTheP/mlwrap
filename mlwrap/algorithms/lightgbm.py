"""LightGBM Wrapper"""

from mlwrap.config import MLConfig
from mlwrap.enums import ProblemType, AlgorithmType


def get_lightgbm(config: MLConfig):
    import lightgbm as lgb

    boosting_type = None
    if config.algorithm_type == AlgorithmType.LightGBMDecisionTree:
        boosting_type = "gbdt"
    elif config.algorithm_type == AlgorithmType.LightGBMRandomForest:
        boosting_type = "rf"

    if config.problem_type == ProblemType.Regression:
        return lgb.LGBMRegressor(
            boosting_type=boosting_type,
            num_iterations=config.maximum_training_iterations,
            num_leaves=config.maximum_tree_leaves,
            max_depth=config.maximum_tree_depth,
            bagging_freq=config.model_training_bagging_frequency,
            bagging_fraction=config.model_training_bagging_fraction,
        )

    if config.problem_type == ProblemType.Classification:
        class_weight = "balanced" if config.adapt_class_weights else None
        return lgb.LGBMClassifier(
            boosting_type=boosting_type,
            class_weight=class_weight,
            num_iterations=config.maximum_training_iterations,
            num_leaves=config.maximum_tree_leaves,
            max_depth=config.maximum_tree_depth,
            bagging_freq=config.model_training_bagging_frequency,
            bagging_fraction=config.model_training_bagging_fraction,
        )
    else:
        raise NotImplementedError
