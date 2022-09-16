"""LightGBM Wrapper"""

from mlwrap.enums import ProblemType

num_iterations: int = 1000
num_leaves: int = 31
max_depth: int = -1
bagging_freq: float = 10
bagging_fraction: int = 0.5

def get_lightgbm(problem_type: ProblemType, boosting_type = "gbdt", adapt_class_weights: bool = False):
    import lightgbm as lgb

    if problem_type == ProblemType.Regression:
        return lgb.LGBMRegressor(
            boosting_type=boosting_type,
            num_iterations=num_iterations,
            num_leaves=num_leaves,
            max_depth=max_depth,
            bagging_freq=bagging_freq,
            bagging_fraction=bagging_fraction,
        )

    if problem_type == ProblemType.Classification:
        class_weight = "balanced" if adapt_class_weights else None
        return lgb.LGBMClassifier(
            boosting_type=boosting_type,
            class_weight=class_weight,
            num_iterations=num_iterations,
            num_leaves=num_leaves,
            max_depth=max_depth,
            bagging_freq=bagging_freq,
            bagging_fraction=bagging_fraction,
        )
    
    raise NotImplementedError
