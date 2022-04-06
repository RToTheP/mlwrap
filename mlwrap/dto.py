from typing import Dict, List, Type, Union

import numpy as np
import pandas as pd
from mlwrap.data.encoders import EncodedFeatureIndex, EncoderBase

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
        encoders: Dict[str, EncoderBase] = None,
        train_input: Union[pd.DataFrame, np.ndarray] = None,
        train_output: Union[pd.DataFrame, np.ndarray] = None,
        test_input: Union[pd.DataFrame, np.ndarray] = None,
        test_output: Union[pd.DataFrame, np.ndarray] = None,
        total_row_count: int = None,        
        encoded_feature_indices: List[Type[EncodedFeatureIndex]] = None,
    ) -> None:
        self.encoders = encoders
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output
        self.total_row_count = total_row_count
        self.encoded_feature_indices = encoded_feature_indices


class Feature:
    def __init__(
        self,
        id: str,
        feature_type: FeatureType,
        active: bool = True,
        encoder_type: EncoderType = None,
        handle_unknown: bool = True,
    ) -> None:
        self.id = id
        self.feature_type = feature_type
        self.active = active
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
        algorithm_type: AlgorithmType = None,
        features: List[Type[Feature]] = None,
        model_feature_id: str = None,
        input_data: InputData = None,
        train_test_split: float = 0.8,
        shuffle_before_splitting: bool = True,
        balance_data_via_resampling: bool = False,
    ) -> None:
        self.algorithm_type = (
            algorithm_type if algorithm_type is not None else AlgorithmType.LightGBMDecisionTree
        )
        self.features = features if features is not None else list()
        self.model_feature_id = model_feature_id
        self.input_data = input_data
        self.train_test_split = train_test_split
        self.shuffle_before_splitting = shuffle_before_splitting
        self.balance_data_via_resampling = balance_data_via_resampling

    def get_model_feature(self):
        return [f for f in self.features if f.id == self.model_feature_id][0]

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