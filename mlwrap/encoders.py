import abc
from collections import Counter
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
from mlwrap.config import Feature, MLConfig
from mlwrap.enums import EncoderType, FeatureType, ProblemType


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
    for feature_id in X.columns:
        column_data = utils.to_numpy(X[feature_id]).reshape(-1, 1)
        if feature_id in config.features:
            feature = config.features[feature_id]
        elif utils.is_categorical(X.dtypes[feature_id]):
            feature = Feature(feature_id, feature_type=FeatureType.Categorical)
        else:
            feature = Feature(feature_id, feature_type=FeatureType.Continuous)

        encoder = get_encoder(feature, column_data)
        pipeline = Pipeline(
            steps=[
                (
                    "1dTo2d",
                    Convert1dTo2d(),
                ),
                ("encoder", encoder),
            ]
        )
        transformers.append((feature_id, pipeline, feature_id))

    return ColumnTransformer(transformers=transformers)


def get_model_feature_encoder(config: MLConfig, y: pd.Series) -> Pipeline:
    column_data = utils.to_numpy(y).reshape(-1, 1)

    if config.model_feature_id in config.features:
        feature = config.features[config.model_feature_id]
    elif utils.is_categorical(y.dtype):
        feature = Feature(config.model_feature_id, feature_type=FeatureType.Categorical)
    else:
        feature = Feature(config.model_feature_id, feature_type=FeatureType.Continuous)
    encoder = get_encoder(feature, column_data)
    return Pipeline(
        steps=[
            (
                "1dTo2d",
                Convert1dTo2d(),
            ),
            (
                "model_feature_encoder",
                encoder,
            ),
            ("flatten", FlattenOutputs(config.problem_type)),
        ]
    )


def get_encoder(feature: Feature, column_data):
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
        handle_unknown_ = "ignore" if feature.handle_unknown else "error"
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
