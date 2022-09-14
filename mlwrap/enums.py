from enum import Enum, unique


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
class HandleUnknown(str, Enum):
    allow = "allow"
    remove = "remove"
