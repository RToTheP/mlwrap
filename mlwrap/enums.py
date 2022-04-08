from enum import Enum, unique


@unique
class Status(str, Enum):
    success = "success"
    success_with_unknown = "success_with_unknown"
    failed = "failed"
    failed_invalid_row = "failed_invalid_row"
    cancelled = "cancelled"
    invalid_data = "invalid_data"
    model_feature_removed = "model_feature_removed"
    model_feature_count_too_low = "model_feature_count_too_low"
    no_valid_rows = "no_valid_rows"


@unique
class ProblemType(str, Enum):
    Classification = "Classification"
    Regression = "Regression"


@unique
class FeatureType(str, Enum):
    Categorical = "Categorical"
    Continuous = "Continuous"
    Text = "Text"


@unique
class DataType(str, Enum):
    Csv = "Csv"
    DataFrame = "DataFrame"


@unique
class EncoderType(str, Enum):
    OneHot = "OneHot"
    MinMax = "MinMax"
    Hash = "Hash"
    Cyclical = "Cyclical"
    Tfidf = "Tfidf"


@unique
class AlgorithmType(str, Enum):
    SklearnLinearModel = "SklearnLinearModel"
    SklearnDecisionTree = "SklearnDecisionTree"
    KerasNeuralNetwork = "KerasNeuralNetwork"
    LightGBMDecisionTree = "LightGBMDecisionTree"
    LightGBMRandomForest = "LightGBMRandomForest"


@unique
class ExplainerType(str, Enum):
    LightGBM = "LightGBM"
    TreeSHAP = "TreeSHAP"
    GradientSHAP = "GradientSHAP"
    LinearSHAP = "LinearSHAP"
    SklearnDecisionTree = "SklearnDecisionTree"
    SklearnLinearModel = "SklearnLinearModel"


@unique
class CleaningType(str, Enum):
    row_feature_out_of_range = "row_feature_out_of_range"
    label_counts_too_low = "label_counts_too_low"
    feature_non_predictive = "feature_non_predictive"
    label_regrouped = "label_regrouped"


@unique
class ScoreType(str, Enum):
    iterations = "iterations"
    total_row_count = "total_row_count"
    active_feature_count = "active_feature_count"
    inactive_feature_count = "inactive_feature_count"
    majority_class_fraction = "majority_class_fraction"
    minority_class_fraction = "minority_class_fraction"
    recall_macro = "recall_macro"
    recall_weighted = "recall_weighted"
    precision_macro = "precision_macro"
    precision_weighted = "precision_weighted"
    f1_macro = "f1_macro"
    f1_weighted = "f1_weighted"
    roc_auc_macro = "roc_auc_macro"
    roc_auc_weighted = "roc_auc_weighted"
    pr_auc_macro = "pr_auc_macro"
    pr_auc_weighted = "pr_auc_weighted"
    mean_absolute_error = "mean_absolute_error"
    median_absolute_error = "median_absolute_error"
    mean_squared_error = "mean_squared_error"


@unique
class HandleUnknown(str, Enum):
    allow = "allow"
    remove = "remove"
