from enum import Enum, unique


@unique
class ProblemType(str, Enum):
    Classification = "Classification"
    Regression = "Regression"
