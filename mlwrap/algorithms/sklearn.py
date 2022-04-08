"""Module for Sklearn algorithms"""

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlwrap.algorithms.base import AlgorithmBase
from mlwrap.config import TrainingResults
from mlwrap.data.config import DataDetails
from mlwrap.data.preparation import flatten_output_data
from mlwrap.enums import AlgorithmType, ProblemType
from mlwrap.explainers import explain_model
from mlwrap.scores import calculate_scores


class SklearnLinearModel(AlgorithmBase):
    @property
    def algorithm(self) -> AlgorithmType:
        return AlgorithmType.SklearnLinearModel

    def _build_model(self):
        problem_type = self._config.problem_type
        logging.info(f"{problem_type} model")
        if problem_type == ProblemType.Regression:
            self._model = LinearRegression()
        elif problem_type == ProblemType.Classification:
            class_weight = "balanced" if self._config.adapt_class_weights else None
            self._model = LogisticRegression(
                max_iter=self._config.maximum_training_iterations,
                class_weight=class_weight,
            )
        else:
            raise NotImplementedError

    def fit(self, data_details: DataDetails) -> TrainingResults:
        if self._model is None:
            self._build_model()

        # flatten the output data
        train_output_flat, _ = flatten_output_data(
            self._config.problem_type, data_details.train_output
        )

        self._model.fit(data_details.train_input, train_output_flat)

        if self._config.problem_type == ProblemType.Classification:
            self.iterations_ = self._model.n_iter_.max()
        elif self._config.problem_type == ProblemType.Regression:
            self.iterations_ = 1  # doesn't mean anything for linear regression

        return self.get_training_results(data_details=data_details)

    def predict(self, data):
        if self._config.problem_type == ProblemType.Classification:
            return self._model.predict_proba(data)
        elif self._config.problem_type == ProblemType.Regression:
            predictions = self._model.predict(data)
            return np.reshape(predictions, (-1, 1))


class SklearnDecisionTree(AlgorithmBase):
    @property
    def algorithm(self) -> AlgorithmType:
        return AlgorithmType.SklearnDecisionTree

    def _build_model(self):
        logging.info(f"{self._config.problem_type} model")
        if self._config.problem_type == ProblemType.Regression:
            self._model = DecisionTreeRegressor()
        elif self._config.problem_type == ProblemType.Classification:
            class_weight = "balanced" if self._config.adapt_class_weights else None
            self._model = DecisionTreeClassifier(class_weight=class_weight)
        else:
            raise NotImplementedError

    def fit(self, data_details: DataDetails) -> TrainingResults:
        if self._model is None:
            self._build_model()

        # flatten the output data
        train_output_flat, _ = flatten_output_data(
            self._config.problem_type, data_details.train_output
        )

        self._model.fit(data_details.train_input, train_output_flat)
        self.iterations_ = 1

        return self.get_training_results(data_details=data_details)

    def predict(self, data):
        if self._config.problem_type == ProblemType.Classification:
            return self._model.predict_proba(data)
        elif self._config.problem_type == ProblemType.Regression:
            predictions = self._model.predict(data)
            return np.reshape(predictions, (-1, 1))
