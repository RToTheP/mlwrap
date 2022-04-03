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
class Status(str, Enum):
    success = "success"
    failed = "failed"


@unique
class AlgorithmType(str, Enum):
    SklearnLinearModel = "SklearnLinearModel"
    SklearnDecisionTree = "SklearnDecisionTree"
    KerasNeuralNetwork = "KerasNeuralNetwork"
    LightGBMDecisionTree = "LightGBMDecisionTree"
    LightGBMRandomForest = "LightGBMRandomForest"
