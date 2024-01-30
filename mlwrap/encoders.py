import abc
import math

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from mlwrap import utils
from mlwrap.config import MLConfig
from mlwrap.enums import ProblemType


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


class FeatureHasherWrapper(FeatureHasher):
    def transform(self, raw_X):
        Xt = super().transform(raw_X)
        return Xt.toarray()


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


class FlattenOutputs(EncoderBase, TransformerMixin):
    def __init__(self, problem_type: ProblemType) -> None:
        self.problem_type = problem_type
        self.fit_shape = None

    def fit(self, X, y):
        self.fit_shape = X.shape[1]
        return self

    def transform(self, X):
        if self.problem_type == ProblemType.Classification:
            # classification problem so flatten the output array
            Xt = np.argmax(X, axis=1)
        elif self.problem_type == ProblemType.Regression:
            Xt = np.squeeze(X, axis=1)
        return Xt

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        Xt = self.transform(X)
        return Xt

    def inverse_transform(self, X):
        if self.problem_type == ProblemType.Classification:
            Xt = np.eye(self.fit_shape)[X]
        elif self.problem_type == ProblemType.Regression:
            Xt = X.reshape(-1, 1)
        return Xt


class Convert1dTo2d(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.ndim == 1:
            return utils.to_numpy(X).reshape(-1, 1)

    def inverse_transform(self, X):
        if X.ndim == 2 and X.shape[1] == 1:
            X = utils.to_numpy(X).flatten()
        return X


def get_column_transformer(config: MLConfig, X: pd.DataFrame) -> ColumnTransformer:
    transformers = []
    # columns in training data
    for c in X.columns:
        if config.encoders is not None and c in config.encoders:
            encoder = config.encoders[c]
        else:
            encoder = get_encoder(X[c])
        pipeline = Pipeline(
            steps=[
                (
                    "1dTo2d",
                    Convert1dTo2d(),
                ),
                ("encoder", encoder),
            ]
        )
        transformers.append((c, pipeline, c))

    return ColumnTransformer(transformers=transformers)


def get_model_feature_encoder(config: MLConfig, y: pd.Series) -> Pipeline:
    if config.encoders is not None and config.model_feature_id in config.encoders:
        encoder = config.encoders[config.model_feature_id]
    else:
        encoder = get_encoder(y)
    return Pipeline(
        steps=[
            (
                "1dTo2d",
                Convert1dTo2d(),
            ),
            (
                "encoder",
                encoder,
            ),
            ("flatten", FlattenOutputs(config.problem_type)),
        ]
    )

def get_one_hot_encoder():
    return OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist", min_frequency=0.1, max_categories=10)

def get_min_max_scaler():
    return MinMaxScaler()

def get_tfidf_encoder(max_features: int = 10000):
    return TfidfEncoder(max_features)

def get_cyclical_encoder(cyclical_period: float):
    return CyclicalEncoder(cyclical_period)

def get_hash_encoder(column_data, hash_size_ratio):
    label_count = np.unique(column_data)
    hash_size = int(round(label_count * float(hash_size_ratio)))
    return FeatureHasherWrapper(n_features=hash_size, input_type="string")


def get_encoder(column_data):
    # If no encoder is set then default to one based on the data type
    if utils.is_categorical(column_data.dtype):
        return get_one_hot_encoder()
    return get_min_max_scaler()
