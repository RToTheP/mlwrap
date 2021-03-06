import abc
from threading import Event
from typing import List, Type, Union
import uuid
import datetime
import time
import logging

import numpy as np
import pandas as pd

from mlwrap.config import (
    ExplanationResult,
    InferenceOutput,
    InferenceResult,
    MLConfig,
    TrainingResults,
)
from mlwrap.data.config import DataDetails
from mlwrap.enums import (
    AlgorithmType,
    EncoderType,
    Status,
)
from mlwrap.explainers import get_explainer, explain_model
from mlwrap.explainers.base import ExplainerBase

from mlwrap.io import save_pkl_bytes, load_pkl
from mlwrap.scores import calculate_scores


class AlgorithmBase(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "fit")
            and callable(subclass.fit)
            and hasattr(subclass, "predict")
            and callable(subclass.predict)
            and hasattr(subclass, "load")
            and callable(subclass.load)
            or NotImplemented
        )

    @property
    @abc.abstractmethod
    def algorithm(self) -> AlgorithmType:
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, data_details: DataDetails) -> TrainingResults:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, data: Union[pd.DataFrame, np.ndarray]):
        raise NotImplementedError

    def __init__(self, config: MLConfig, stop_event: Event):
        self._config: MLConfig = config
        self._model = None
        self.id: str = uuid.uuid1().hex
        self.iterations_: int = 0
        self._stop_event: Event = stop_event

    def infer(self, data_details: DataDetails) -> InferenceOutput:
        logging.debug(f"infer - start: {datetime.datetime.now()}")

        t_start = time.perf_counter()
        predictions = self.predict(data_details.inference_input)

        logging.debug(f"predict(): {time.perf_counter() - t_start:0.4f} seconds")

        # explain the inferences
        explanation_results: List[Type[ExplanationResult]] = [None] * len(predictions)
        if self._config.explain:
            explainer: ExplainerBase = get_explainer(
                config=self._config, algorithm=self
            )
            explanation_results = explainer.explain(data_details=data_details)

        # get the inference results
        inference_results: List[Type[InferenceResult]] = []

        # then loop over the predictions
        for inference_id, prediction, explanation_result in zip(
            data_details.inference_ids, predictions, explanation_results
        ):
            hit_unknown = inference_id in data_details.unknown_values_input_rows
            status = Status.success_with_unknown if hit_unknown else Status.success

            # decode the inference results
            model_feature = self._config.features[self._config.model_feature_id]
            encoder = data_details.encoders[model_feature.id]
            prediction_inverse_transform = (
                np.reshape(prediction, (1, -1)) if prediction.ndim == 1 else prediction
            )
            result = (
                encoder.inverse_transform(prediction_inverse_transform)
                .flatten()
                .tolist()
            )

            probabilities = None
            labels = None
            if model_feature.encoder_type == EncoderType.OneHot:
                probabilities = prediction.tolist()
                labels = np.array(encoder.categories_).flatten().tolist()

            inference_result: InferenceResult = InferenceResult(
                inference_id=inference_id,
                feature_id=model_feature.id,
                status=status,
                values=result,
                probabilities=probabilities,
                labels=labels,
                explanation_result=explanation_result,
            )

            inference_results.append(inference_result)
        logging.debug(f"infer - end: {datetime.datetime.now()}")

        return InferenceOutput(
            status=data_details.status,
            cleaning_report=data_details.cleaning_report,
            inference_results=inference_results,
        )

    def get_training_results(self, data_details: DataDetails) -> TrainingResults:
        scores = calculate_scores(
            self._config, self.iterations_, self.predict, data_details
        )

        # explain the model
        if self._config.explain:
            explanation_result = explain_model(
                config=self._config, algorithm=self, data_details=data_details
            )
        else:
            explanation_result = None

        # save the models and encoders
        encoder_bytes = data_details.get_encoder_bytes()
        model_bytes = self.get_model_bytes()
        background_data_bytes = data_details.get_background_data_bytes()

        # construct the results

        return TrainingResults(
            status=Status.success,
            cleaning_report=data_details.cleaning_report,
            scores=scores,
            model_bytes=model_bytes,
            encoder_bytes=encoder_bytes,
            explanation_result=explanation_result,
            background_data_bytes=background_data_bytes,
        )

    def get_model_bytes(self) -> bytes:
        return save_pkl_bytes(self._model)

    def load_model(self, model_bytes: bytes):
        return load_pkl(model_bytes)

    def load(self) -> bool:
        self._model = self.load_model(self._config.model_bytes)
        if self._model is None:
            return False
        return True
