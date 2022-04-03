from typing import List, Type, Union

import numpy as np
import pandas as pd

from mlwrap.enums import (
    AlgorithmType,
    DataType,
    EncoderType,
    FeatureType,
    ProblemType,
    Status,
)


class DataDetails:
    def __init__(
        self,
        train_input: Union[pd.DataFrame, np.ndarray] = None,
        train_output: Union[pd.DataFrame, np.ndarray] = None,
        test_input: Union[pd.DataFrame, np.ndarray] = None,
        test_output: Union[pd.DataFrame, np.ndarray] = None,
    ) -> None:
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output


class Feature:
    def __init__(
        self,
        id: str,
        feature_type: FeatureType,
        encoder_type: EncoderType = None,
        handle_unknown: bool = True,
    ) -> None:
        self.id = id
        self.feature_type = feature_type
        self.encoder_type = encoder_type
        self.handle_unknown = handle_unknown


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


class MLSettings:
    def __init__(
        self,
        algorithm: AlgorithmType = None,
        features: List[Type[Feature]] = None,
        model_feature_id: str = None,
        input_data: InputData = None,
        train_test_split: float = 0.8,
        shuffle_before_splitting: bool = True,
    ) -> None:
        self.algorithm = (
            algorithm if algorithm is not None else AlgorithmType.LightGBMDecisionTree
        )
        self.features = features if features is not None else list()
        self.model_feature_id = model_feature_id
        self.input_data = input_data
        self.train_test_split = train_test_split
        self.shuffle_before_splitting = shuffle_before_splitting


    def get_model_feature(self):
        return [
            f for f in self.features if f.id == self.model_feature_id
        ][0]

    def get_problem_type(self):
        model_feature = self.get_model_feature()
        return (
            ProblemType.Classification
            if model_feature.feature_type == FeatureType.Categorical
            else ProblemType.Regression
        )

class TrainingResults:
    def __init__(self, status: Status) -> None:
        self.status = status
