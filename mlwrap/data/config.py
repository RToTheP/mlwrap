from typing import Dict, List, Type, Union

import numpy as np
import pandas as pd
from mlwrap.config import CleaningReport

from mlwrap.data.encoders import EncodedFeatureIndex, EncoderBase
from mlwrap.enums import Status
from mlwrap.io import save_pkl_bytes


class DataDetails:
    def __init__(
        self,
        status: Status,
        cleaning_report: CleaningReport = None,
        encoders: Dict[str, EncoderBase] = None,
        train_input: Union[pd.DataFrame, np.ndarray] = None,
        train_output: Union[pd.DataFrame, np.ndarray] = None,
        test_input: Union[pd.DataFrame, np.ndarray] = None,
        test_output: Union[pd.DataFrame, np.ndarray] = None,
        class_weights: np.ndarray = None,
        class_ratios: Dict[str, float] = None,
        total_row_count: int = None,
        encoded_feature_indices: List[Type[EncodedFeatureIndex]] = None,
        background_data: Union[pd.DataFrame, np.ndarray] = None,
    ) -> None:
        self.status = status
        self.cleaning_report = cleaning_report
        self.encoders = encoders
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output
        self.class_weights = class_weights
        self.class_ratios = class_ratios
        self.total_row_count = total_row_count
        self.encoded_feature_indices = encoded_feature_indices
        self.background_data = background_data

    def get_encoder_bytes(self) -> bytes:
        return save_pkl_bytes(self.encoders)

    def get_background_data_bytes(self) -> bytes:
        return save_pkl_bytes(self.background_data)
