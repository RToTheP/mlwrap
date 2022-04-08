import abc
from collections import Counter
import math
from typing import Dict, List, Tuple, Type, Union


import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from mlwrap.config import Feature, MLConfig
from mlwrap.enums import EncoderType, FeatureType


class EncoderBase(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "fit")
            and callable(subclass.fit)
            and hasattr(subclass, "transform")
            and callable(subclass.transform)
            and hasattr(subclass, "inverse_transform")
            and callable(subclass.inverse_transform)
            or NotImplemented
        )


class CyclicalEncoder(EncoderBase):
    def __init__(self, cyclical_period):
        self.cyclical_period = cyclical_period

    def fit(self, data):
        return

    def transform(self, data):
        sin_data = np.sin(2 * np.pi * data.astype(float) / self.cyclical_period)
        cos_data = np.cos(2 * np.pi * data.astype(float) / self.cyclical_period)
        stacked_data = np.column_stack((sin_data, cos_data))
        return stacked_data

    def inverse_transform(self, data):
        # assume columns are sin,cos
        inverse_data = []
        for row in data:
            inverse_data.append(self._inverse_transform(row))
        inv_data = np.array(inverse_data)
        inv_data = inv_data.reshape(-1, 1)
        return inv_data

    def _inverse_transform(self, row):
        sin_th = row[0]
        cos_th = row[1]
        tan_th = sin_th / cos_th
        inv = (self.cyclical_period / 2 / np.pi) * math.atan(tan_th)
        if sin_th < 0:
            inv = inv + self.cyclical_period
        if cos_th < 0:
            inv = inv + math.copysign(self.cyclical_period / 2, sin_th)
        return inv


class FeatureHasherWrapper(EncoderBase):
    def __init__(self, feature_hasher):
        self.feature_hasher = feature_hasher

    def fit(self, data):
        return

    def transform(self, data):
        hashed_data = self.feature_hasher.transform(data)
        return hashed_data.toarray()


class TfidfEncoder(EncoderBase):
    def __init__(self, max_features: int = 10000):
        self.max_features = max_features
        self.encoder = TfidfVectorizer(max_features=self.max_features)

    def fit(self, data):
        # Reshape the data from (N,1) -> (N)
        self.encoder.fit(
            data.reshape(
                -1,
            )
        )
        return

    def transform(self, data):
        # Reshape the data from (N,1) -> (N)
        transformed_array = self.encoder.transform(
            data.reshape(
                -1,
            )
        ).toarray()
        transformed_array = pd.DataFrame(transformed_array)
        transformed_array = transformed_array.add_prefix("feature_")
        return transformed_array

    def inverse_transform(self, data):
        return


class EncodedFeatureIndex:
    def __init__(self, feature_id: str, start_index: int, index_count: int) -> None:
        self.feature_id = feature_id
        self.start_index = start_index
        self.index_count = index_count


class ColumnTransformer:
    def __init__(self) -> None:
        self.start_index: int = 0
        self.encoded_feature_indices: List[Type[EncodedFeatureIndex]] = list()

    def transform_column(
        self, feature: Feature, column_data: np.ndarray, encoder: EncoderBase
    ) -> np.ndarray:

        transformed_data = encoder.transform(column_data)
        n_columns = transformed_data.shape[1]
        self.encoded_feature_indices.append(
            EncodedFeatureIndex(feature.id, self.start_index, n_columns)
        )
        self.start_index = self.start_index + n_columns
        return transformed_data


def transform(
    data: Union[pd.DataFrame, pd.Series],
    config: MLConfig,
    encoders: Dict[str, EncoderBase],
) -> Tuple[np.ndarray, List[Type[EncodedFeatureIndex]]]:
    if data is None:
        raise ValueError("data is missing")

    feature_dct = {feature.id: feature for feature in config.features}
    features = []
    columnData = []
    encodersList = []

    if isinstance(data, pd.Series):
        features.append(feature_dct[data.name])
        columnData.append(data.to_numpy(dtype=str).reshape(-1, 1))
        encodersList.append(encoders[data.name])
    elif isinstance(data, pd.DataFrame):
        for column in data.columns:
            features.append(feature_dct[column])
            columnData.append(data[column].to_numpy(dtype=str).reshape(-1, 1))
            encodersList.append(encoders[column])

    column_transformer: ColumnTransformer = ColumnTransformer()
    transformed_data = np.column_stack(
        list(
            map(column_transformer.transform_column, features, columnData, encodersList)
        )
    ).astype(np.float32)

    return transformed_data, column_transformer.encoded_feature_indices


def get_fitted_encoders(data: pd.DataFrame, config: MLConfig) -> Dict[str, EncoderBase]:
    if data is None:
        raise ValueError("data is missing")

    features = {feature.id: feature for feature in config.features}
    encoders = {}
    for column in data.columns:
        feature = features[column]
        column_data = data[feature.id].to_numpy(dtype=str).reshape(-1, 1)
        encoder = _get_encoder(feature, column_data)
        encoder.fit(column_data)
        encoders[column] = encoder

    return encoders


def _get_encoder(feature: Feature, column_data):
    # If no encoder is set then default to one based on the data type
    if feature.encoder_type is None:
        if feature.feature_type == FeatureType.Categorical:
            feature.encoder_type = EncoderType.OneHot
        elif feature.feature_type == FeatureType.Continuous:
            feature.encoder_type = EncoderType.MinMax
        elif feature.feature_type == FeatureType.Text:
            feature.encoder_type = EncoderType.Tfidf
        else:
            raise NotImplementedError(
                f"Unsupported feature type {feature.feature_type}"
            )

    if feature.encoder_type == EncoderType.OneHot:
        handle_unknown_ = "ignore" if feature.handle_unknown == True else "error"
        return OneHotEncoder(sparse=False, handle_unknown=handle_unknown_)
    elif feature.encoder_type == EncoderType.MinMax:
        return MinMaxScaler()
    elif feature.encoder_type == EncoderType.Tfidf:
        return TfidfEncoder(feature.max_features)
    elif feature.encoder_type == EncoderType.Cyclical:
        return CyclicalEncoder(float(feature.cyclical_period))
    elif feature.encoder_type == EncoderType.Hash:
        label_count = len(Counter(column_data.flatten()))
        hash_size = int(round(label_count * float(feature.hash_size_ratio)))
        return FeatureHasherWrapper(
            FeatureHasher(n_features=hash_size, input_type="string")
        )
    else:
        raise NotImplementedError(f"Unsupported encoder type: {feature.encoder}")
