from enum import Enum, unique


@unique
class ProblemType(str, Enum):
    Classification = "Classification"
    Regression = "Regression"


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
