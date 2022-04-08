"""LightGBM Wrapper"""

import logging
from threading import Event


import lightgbm as lgb
import numpy as np

from mlwrap.algorithms.base import AlgorithmBase
from mlwrap.config import MLConfig, TrainingResults
from mlwrap.data.config import DataDetails
from mlwrap.data.preparation import get_validation_data, flatten_output_data
from mlwrap.enums import ProblemType, AlgorithmType
from mlwrap.explainers import explain_model


class LightGBMWrapper(AlgorithmBase):
    @property
    def algorithm(self) -> AlgorithmType:
        return self._alg_name

    def __init__(
        self,
        config: MLConfig,
        stop_event: Event,
        alg_name: AlgorithmType,
    ):
        super().__init__(config, stop_event)
        self._alg_name = AlgorithmType[alg_name]
        if alg_name == AlgorithmType.LightGBMDecisionTree:
            self._boosting_type = "gbdt"
        elif alg_name == AlgorithmType.LightGBMRandomForest:
            self._boosting_type = "rf"
        else:
            self._boosting_type = None

    def lightgbm_callback(self):
        def callback(env):
            callback.iteration = env.iteration + 1
            if callback.stop_event is not None and callback.stop_event.is_set():
                logging.info("Stopping due to cancellation event")
                raise lgb.callback.EarlyStopException(env.iteration, 0)

        callback.before_iteration = True
        callback.order = 0
        callback.stop_event = self._stop_event
        return callback

    def fit(self, data_details: DataDetails) -> TrainingResults:
        if self._model is None:
            self._build_model()

        # flatten the output data
        train_output_flat, test_output_flat = flatten_output_data(
            self._config.problem_type,
            data_details.train_output,
            data_details.test_output,
        )

        # set up eval data and possibly have it as None
        validation_data = get_validation_data(data_details.test_input, test_output_flat)

        # fit the model
        cb = self.lightgbm_callback()
        self._model.fit(
            data_details.train_input,
            train_output_flat,
            verbose=True,
            eval_set=[validation_data],
            early_stopping_rounds=self._config.early_stopping_iterations,
            callbacks=[cb],
        )
        self.iterations_ = cb.iteration

        return self.get_training_results(data_details=data_details)

    def _build_model(self):
        logging.info(f"{self._config.problem_type} model")
        if self._config.problem_type == ProblemType.Regression:
            self._model = lgb.LGBMRegressor(
                boosting_type=self._boosting_type,
                num_iterations=self._config.maximum_training_iterations,
                num_leaves=self._config.maximum_tree_leaves,
                max_depth=self._config.maximum_tree_depth,
                bagging_freq=self._config.model_training_bagging_frequency,
                bagging_fraction=self._config.model_training_bagging_fraction,
            )
        elif self._config.problem_type == ProblemType.Classification:
            class_weight = "balanced" if self._config.adapt_class_weights else None
            self._model = lgb.LGBMClassifier(
                boosting_type=self._boosting_type,
                class_weight=class_weight,
                num_iterations=self._config.maximum_training_iterations,
                num_leaves=self._config.maximum_tree_leaves,
                max_depth=self._config.maximum_tree_depth,
                bagging_freq=self._config.model_training_bagging_frequency,
                bagging_fraction=self._config.model_training_bagging_fraction,
            )
        else:
            raise NotImplementedError

    def predict(self, data):
        if self._config.problem_type == ProblemType.Classification:
            return self._model.predict_proba(
                data, verbose=0, num_iteration=self._model.best_iteration_
            )
        elif self._config.problem_type == ProblemType.Regression:
            predictions = self._model.predict(
                data, verbose=0, num_iteration=self._model.best_iteration_
            )
            return np.reshape(predictions, (-1, 1))
